"""Post-test audit: does the spatial CNN rescue or overwrite useful attention?"""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
import torch
from shapely.geometry import box

from clickagent.action_benchmark import sha256,write_json
from clickagent.conditioned_grounder import load_spatial_checkpoint,make_batch
from clickagent.spatial_geometry import decode_candidates,shape_from_geometry
from run_target_token_experiment import read_json,read_jsonl
from train_conditioned_readouts import load_records,batches


def intersects_target(index,grid,target):
    gh,gw=grid;row,col=divmod(int(index),gw)
    return target.intersection(box(col/gw,row/gh,(col+1)/gw,(row+1)/gh)).area>0


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=Path('runs/conditioned_v3'))
    args=p.parse_args();out=args.output;torch.set_num_threads(4)
    freeze=read_json(out/'frozen_readouts.json');checkpoint=out/'models/full.pt'
    assert sha256(checkpoint)==freeze['models']['full']['sha256']
    cases=[c for c in read_jsonl(out/'manifest.jsonl') if c['experiment_split']=='test']
    cached={r['sample_id']:r for r in read_jsonl(out/'evaluation/predictions.jsonl') if r['method']=='full'}
    records,_=load_records(out,cases,with_labels=False)
    model,_=load_spatial_checkpoint(checkpoint,'cpu');rows=[];counts=Counter();matrix=Counter()
    with torch.inference_mode():
        for group in batches(records,8):
            inputs,_=make_batch(group,'cpu',labels=False);result=model(**inputs);gh,gw=inputs['grid_shape']
            pooled=torch.einsum('bt,bthp->bhp',result['token_weights'],inputs['attention'].float())
            mass=pooled.sum(-1)
            relative=torch.log1p((gh*gw)*pooled/mass.unsqueeze(-1).clamp_min(1e-12))
            prior=torch.einsum('bh,bhp->bp',result['head_weights'],relative).reshape(-1,gh,gw)
            residual=result['logits']-prior.repeat_interleave(2,-2).repeat_interleave(2,-1)
            for i,r in enumerate(group):
                c=r['case'];target=shape_from_geometry(c['target_geometry']);reference=cached[c['sample_id']]
                p_index=int(prior[i].argmax());raw_index=int(r['target_attention'][0,:32].mean(0).argmax())
                p_hit=intersects_target(p_index,[gh,gw],target);raw_hit=intersects_target(raw_index,[gh,gw],target)
                proposal=decode_candidates(result['logits'][i].numpy(),result['offsets'][i].numpy(),result['grid_shape'],c['image_size'],top_k=1)[0]
                f_hit=reference['selected_coarse_cell_overlap'];cell=proposal['patch_index']
                fine_row,fine_col=divmod(cell,gw*2);p_row,p_col=divmod(p_index,gw)
                same_parent=(fine_row//2==p_row and fine_col//2==p_col)
                counts['raw_layer15_coarse_overlap']+=raw_hit;counts['learned_attention_prior_coarse_overlap']+=p_hit
                counts['final_coarse_overlap']+=f_hit;counts['same_coarse_argmax_before_after_cnn']+=same_parent
                counts['cpu_prediction_matches_cached_gpu']+=proposal['point']==reference['point']
                matrix[f'prior_{int(p_hit)}_final_{int(f_hit)}']+=1
                rows.append({'sample_id':c['sample_id'],'dataset':c['dataset'],'generated_target_name':reference['generated_target_name'],
                    'raw_layer15_peak_cell':raw_index,'raw_layer15_cell_overlap':raw_hit,
                    'learned_prior_peak_cell':p_index,'learned_prior_cell_overlap':p_hit,'final_coarse_cell_overlap':f_hit,
                    'final_point':reference['point'],'cpu_point':proposal['point'],'strict_hit':reference['hit'],
                    'cnn_residual_at_final_peak':float(residual[i,fine_row,fine_col]),
                    'attention_prior_at_final_peak':float(prior[i,fine_row//2,fine_col//2]),
                    'cnn_residual_at_prior_peak':float(residual[i,p_row*2,p_col*2])})
    result={'n_test':len(cases),'named':len(rows),'counts':dict(counts),'prior_vs_final_coarse_matrix':dict(matrix),
        'source_sha256':sha256(Path(__file__)),'checkpoint_sha256':sha256(checkpoint),
        'posthoc_diagnostic_only':True,'no_weights_or_primary_predictions_modified':True,
        'note':'Cell overlap is not click accuracy. The prior is jointly trained with the CNN; this decomposition is explanatory and is not an independently selected model.'}
    write_json(out/'evaluation/fusion_diagnostics.json',result)
    (out/'evaluation/fusion_cases.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
