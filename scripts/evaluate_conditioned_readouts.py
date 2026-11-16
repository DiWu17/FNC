"""One sealed test for all frozen spatial readouts and matched baselines."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
import torch

from clickagent.action_benchmark import sha256,write_json
from clickagent.action_grounding import AttentionPointGrounder
from clickagent.conditioned_grounder import load_spatial_checkpoint,make_batch
from clickagent.public_evaluation import point_in_geometry,quantization_oracle
from clickagent.spatial_geometry import decode_candidates,shape_from_geometry
from clickagent.target_token import spatial_scores
from run_target_token_experiment import read_json,read_jsonl,key
from train_conditioned_readouts import load_records,batches


def score_point(point,case):
    return bool(point is not None and point_in_geometry([point[0]/case['image_size'][0],point[1]/case['image_size'][1]],case['target_geometry']))


def describe_candidates(proposals,case,coarse_grid,fine_grid):
    if not proposals:return {'candidate_point_oracle_hit':False,'selected_coarse_cell_overlap':False,
        'selected_fine_cell_overlap':False,'candidate_region_window_oracle_overlap':False}
    first=proposals[0];row,col=divmod(first['patch_index'],fine_grid[1])
    from shapely.geometry import box
    target=shape_from_geometry(case['target_geometry'])
    gh,gw=coarse_grid;fh,fw=fine_grid
    coarse_row,coarse_col=row*gh//fh,col*gw//fw
    coarse_box=[coarse_col/gw,coarse_row/gh,(coarse_col+1)/gw,(coarse_row+1)/gh]
    return {'candidate_point_oracle_hit':any(score_point(r['point'],case) for r in proposals),
        'selected_coarse_cell_overlap':target.intersection(box(*coarse_box)).area>0,
        'selected_fine_cell_overlap':target.intersection(box(*first['cell'])).area>0,
        'candidate_region_window_oracle_overlap':any(target.intersection(box(*r['region'])).area>0 for r in proposals)}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=Path('runs/conditioned_v3'))
    p.add_argument('--device',default='cuda:0');args=p.parse_args();out=args.output
    freeze=read_json(out/'frozen_readouts.json');assert freeze['test_features_at_freeze']==0
    assert read_json(out/'features_test_completion.json')['complete']
    for name,digest in freeze['head_sources'].items():assert sha256(Path(name))==digest
    cases=[c for c in read_jsonl(out/'manifest.jsonl') if c['experiment_split']=='test'];assert len(cases)==192
    records,_=load_records(out,cases,with_labels=False);rows=[];coarse_oracle=0;fine_oracle=0
    legacy=AttentionPointGrounder.load('runs/action_v3/grounder/grounder.joblib')
    for r in records:
        c=r['case'];meta=r['metadata'];attention=r['target_attention']
        coarse_oracle+=quantization_oracle(c['target_geometry'],meta['grid_shape'],c['image_size'])
        fine_oracle+=quantization_oracle(c['target_geometry'],meta['fine_grid_shape'],c['image_size'])
        with np.load(out/'features'/f"{key(c['sample_id'])}.npz",allow_pickle=False) as z:
            prefill_point=spatial_scores(z['prefill_attention'],'legacy_96',legacy,meta['grid_shape'],c['image_size'])
        rows.append(dict(sample_id=c['sample_id'],dataset=c['dataset'],method='legacy_prefill',point=prefill_point,hit=score_point(prefill_point,c)))
        for name,heat,mode in (('legacy_first',attention[0] if len(attention) else None,'legacy_96'),
                ('legacy_mean',attention.mean(0) if len(attention) else None,'legacy_96'),
                ('layer15_mean',attention[0] if len(attention) else None,'layer15_mean')):
            point=spatial_scores(heat,mode,legacy,meta['grid_shape'],c['image_size']) if heat is not None else None
            rows.append(dict(sample_id=c['sample_id'],dataset=c['dataset'],method=name,point=point,hit=score_point(point,c)))
        native=read_json(out/'native'/f"{key(c['sample_id'])}.json")
        for name,point in (('qwen_native_strict',native['point']),('qwen_native_completed_lax',native['completed_lax_point'])):
            rows.append(dict(sample_id=c['sample_id'],dataset=c['dataset'],method=name,point=point,hit=score_point(point,c)))
    for variant,frozen in freeze['models'].items():
        path=out/'models'/frozen['file'];assert sha256(path)==frozen['sha256']
        model,metadata=load_spatial_checkpoint(path,args.device)
        done=set()
        with torch.inference_mode():
            for group in batches(records,8):
                inputs,_=make_batch(group,args.device,labels=False);result=model(**inputs)
                logits=result['logits'].cpu().numpy();offsets=result['offsets'].cpu().numpy();fine_grid=result['grid_shape']
                for i,r in enumerate(group):
                    c=r['case'];meta=r['metadata'];done.add(c['sample_id'])
                    proposals=decode_candidates(logits[i],offsets[i],fine_grid,c['image_size'],top_k=5)
                    row=dict(sample_id=c['sample_id'],dataset=c['dataset'],method=variant,point=proposals[0]['point'],
                        hit=score_point(proposals[0]['point'],c),candidates=proposals,
                        **describe_candidates(proposals,c,meta['grid_shape'],fine_grid))
                    rows.append(row)
                    if variant=='full':
                        row['generated_target_name']=meta['target_text']
                        row['target_token_weights']=result['token_weights'][i,:len(r['target_attention'])].cpu().tolist()
                        row['dynamic_head_weights']=result['head_weights'][i].cpu().tolist()
                        row['head_image_mass']=result['head_mass'][i].cpu().tolist()
                        row['target_raw_indices']=meta['target_raw_indices']
                        policy_point=proposals[0]['point'] if meta['action']=='CLICK' else None
                        rows.append(dict(sample_id=c['sample_id'],dataset=c['dataset'],method='full_generated_action_policy',
                            point=policy_point,hit=score_point(policy_point,c)))
                        centers=decode_candidates(logits[i],offsets[i],fine_grid,c['image_size'],top_k=5,center_only=True)
                        rows.append(dict(sample_id=c['sample_id'],dataset=c['dataset'],method='full_fine_centers',point=centers[0]['point'],hit=score_point(centers[0]['point'],c)))
                        row_index,col=divmod(proposals[0]['patch_index'],fine_grid[1]);w,h=c['image_size'];gh,gw=meta['grid_shape']
                        point=[min(w-1,round((col//2+.5)*w/gw)),min(h-1,round((row_index//2+.5)*h/gh))]
                        rows.append(dict(sample_id=c['sample_id'],dataset=c['dataset'],method='full_coarse_centers',point=point,hit=score_point(point,c)))
            for r in records:
                c=r['case']
                if c['sample_id'] in done:continue
                rows.append(dict(sample_id=c['sample_id'],dataset=c['dataset'],method=variant,point=None,hit=False,
                    generated_target_name=r['metadata']['target_text'],
                    candidates=[],**describe_candidates([],c,r['metadata']['grid_shape'],r['metadata']['fine_grid_shape'])))
                if variant=='full':
                    for mode in ('full_fine_centers','full_coarse_centers','full_generated_action_policy'):
                        rows.append(dict(sample_id=c['sample_id'],dataset=c['dataset'],method=mode,point=None,hit=False))
        print(f'Scored {variant}: '+str(sum(r['hit'] for r in rows if r['method']==variant)),flush=True)
        del model
        if torch.cuda.is_available():torch.cuda.empty_cache()
    dest=out/'evaluation';dest.mkdir(exist_ok=True)
    (dest/'predictions.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
    methods={}
    for method in sorted({r['method'] for r in rows}):
        group=[r for r in rows if r['method']==method];assert len(group)==len(cases)
        assert {r['sample_id'] for r in group}=={c['sample_id'] for c in cases}
        methods[method]={'n':len(cases),'hits':sum(r['hit'] for r in group),'accuracy':sum(r['hit'] for r in group)/len(cases),
            'datasets':{ds:{'n':sum(c['dataset']==ds for c in cases),'hits':sum(r['hit'] for r in group if r['dataset']==ds)} for ds in sorted({c['dataset'] for c in cases})}}
    summary={'n_test':len(cases),'methods':methods,'primary_method':'full','available_names':sum(r['metadata']['target_available'] for r in records),
        'coarse_center_geometric_ceiling':int(coarse_oracle),'fine_center_geometric_ceiling':int(fine_oracle),
        'all_failures_retained':True,'test_used_for_model_selection':False}
    write_json(dest/'summary.json',summary)
    write_json(dest/'identity.json',{'evaluation_source_sha256':sha256(Path(__file__)),
        'frozen_readouts_sha256':sha256(out/'frozen_readouts.json'),'manifest_sha256':sha256(out/'manifest.jsonl')})
    print(json.dumps(summary,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
