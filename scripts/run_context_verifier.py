"""Label-free v9 visual inference: fixed pools and contextual multi-region pools."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np
from PIL import Image
import torch

from clickagent.action_benchmark import sha256,write_json
from clickagent.context_verifier import ContextFeatureRuntime,compare_candidates,diverse_crops
from clickagent.model_features import ExtractionConfig
from clickagent.point_aligned_inference import proposal_crop,map_crop_point,unique_points,verify_point
from point_aligned_data import archive_record,read
from point_aligned_predict import load_predictors,predict_record
from run_point_aligned_inference import save_features,native
from run_target_token_experiment import key,read_jsonl
from train_context_point_critic import raw_path

OUT=Path('runs/context_verifier_v9')
PRIMARY='joint_connected_s1'


def symmetric_choice(forward,reverse,pool):
    if not pool:return dict(point=None,forced_point=None,scores=[],order_disagrees=False)
    scores=np.zeros(len(pool)+1,dtype=np.float64)
    for result in (forward,reverse):
        values=np.asarray(result['scores'],np.float64);values-=np.logaddexp.reduce(values)
        lookup={tuple(p):v for p,v in zip(result['order'],values[:-1])}
        scores+=np.asarray([lookup[tuple(v['point'])] for v in pool]+[values[-1]])/2
    winner=int(scores.argmax());forced=int(scores[:-1].argmax())
    return dict(point=pool[winner]['point'] if winner<len(pool) else None,
        forced_point=pool[forced]['point'],scores=scores.tolist(),
        order_disagrees=forward['point']!=reverse['point'])


def full_features(runtime,image,case,root,validation):
    path=root/'features'/f'{key(case["sample_id"])}.npz'
    if not path.exists():
        if validation:
            original=raw_path(case);assert original.exists();shutil.copyfile(original,path)
        else:
            arrays,meta=runtime.infer_spatial_features(image,case['instruction'])
            meta.update(sample_id=case['sample_id'],image_sha256=case['image_sha256'],
                instruction_sha256=hashlib.sha256(case['instruction'].encode()).hexdigest())
            save_features(path,arrays,meta)
    record=archive_record(path,case,labels=False)
    assert record['metadata']['sample_id']==case['sample_id']
    assert record['metadata']['image_sha256']==case['image_sha256']
    return record


def local_features(runtime,image,crop,case,root,name,transform,context,target_name):
    sid=key(case['sample_id']);path=root/name/f'{sid}.npz'
    if not path.exists():
        if context:
            arrays,meta=runtime.infer_context_features(image,crop,case['instruction'],transform,target_name)
        else:arrays,meta=runtime.infer_spatial_features(crop,case['instruction'])
        meta.update(sample_id=case['sample_id'],image_sha256=case['image_sha256'],transform=transform,
            instruction_sha256=hashlib.sha256(case['instruction'].encode()).hexdigest())
        save_features(path,arrays,meta)
    local_case=dict(sample_id=case['sample_id'],image_size=list(crop.size),image_sha256=case['image_sha256'])
    record=archive_record(path,local_case,labels=False)
    assert record['metadata']['transform']==transform
    return record


def mapped_candidates(result,transform,source):
    return [dict(v,point=map_crop_point(v['point'],transform),local_point=v['point'],source=source) for v in result['candidates']]


def process(runtime,models,case,root,validation=False):
    sid=key(case['sample_id']);dest=root/'pipeline'/f'{sid}.json'
    if dest.exists():
        result=read(dest)
        stale=any(pool and result['comparisons'][name]['forward'].get('per_candidate_pixel_budget')!=262144
                  for name,pool in result['pools'].items())
        if not stale:return result
        assert validation,'Fresh-test comparison budget must match sealed protocol'
        # Preserve the initial validation-only low-detail interface trial.
        backup=root/'initial_low_detail_comparisons';backup.mkdir(exist_ok=True)
        write_json(backup/f'{sid}.json',result['comparisons'])
        with Image.open(case['image']) as im:image=im.convert('RGB')
        for name,pool in result['pools'].items():
            fwd=compare_candidates(runtime,image,case['instruction'],pool,False)
            rev=compare_candidates(runtime,image,case['instruction'],pool,True)
            comp=dict(forward=fwd,reverse=rev,symmetric=symmetric_choice(fwd,rev,pool))
            write_json(root/'comparison'/f'{sid}_{name}.json',comp);result['comparisons'][name]=comp
        write_json(dest,result);return result
    assert sha256(Path(case['image']))==case['image_sha256']
    with Image.open(case['image']) as im:image=im.convert('RGB')
    record=full_features(runtime,image,case,root,validation)
    path=root/'global'/f'{sid}.json'
    if not path.exists():write_json(path,predict_record(record,models,only=[PRIMARY],top_k=64)[PRIMARY])
    global_result=read(path)
    native(runtime,image,case['instruction'],root/'native'/f'{sid}.json')
    global_candidates=[dict(v,source='global') for v in global_result['candidates']]
    locals=[];old_result=None;old_candidates=[];old_transform=None
    if global_candidates:
        top=global_candidates[0];crop,old_transform=proposal_crop(image,top['parent_index'],record['metadata']['grid_shape'])
        old_record=local_features(runtime,image,crop,case,root,'old_crop_features',old_transform,False,global_result['target_name'])
        old_result=predict_record(old_record,models,only=[PRIMARY])[PRIMARY]
        old_candidates=mapped_candidates(old_result,old_transform,'old_crop')
        proposals=diverse_crops(image,global_candidates,record['metadata']['grid_shape'],maximum=2)
        for i,(crop,transform) in enumerate(proposals):
            name=f'context_features_{i}';local_record=local_features(runtime,image,crop,case,root,name,transform,True,global_result['target_name'])
            result=predict_record(local_record,models,only=[PRIMARY])[PRIMARY]
            locals.append(dict(result=result,transform=transform,candidates=mapped_candidates(result,transform,f'context_{i}')))
    fixed_pool=unique_points(global_candidates[:2],old_candidates[:2],limit=4)
    c0=locals[0]['candidates'] if len(locals)>0 else [];c1=locals[1]['candidates'] if len(locals)>1 else []
    context_pool=unique_points(global_candidates[:1],c0[:1],c1[:1],global_candidates[1:2],c0[1:2],c1[1:2],global_candidates[2:4],limit=4)
    pools={'fixed_pool':fixed_pool,'context_pool':context_pool}
    union=unique_points(fixed_pool,context_pool);verification={}
    for item in union:
        x,y=item['point'];path=root/'verifier'/f'{sid}_{x}_{y}.json'
        if not path.exists():write_json(path,verify_point(runtime,image,case['instruction'],item['point']))
        verification[f'{x},{y}']=read(path)
    comparisons={}
    for name,pool in pools.items():
        path=root/'comparison'/f'{sid}_{name}.json'
        if not path.exists():
            fwd=compare_candidates(runtime,image,case['instruction'],pool,False)
            rev=compare_candidates(runtime,image,case['instruction'],pool,True)
            write_json(path,dict(forward=fwd,reverse=rev,symmetric=symmetric_choice(fwd,rev,pool)))
        comparisons[name]=read(path)
    result=dict(sample_id=case['sample_id'],global_result=global_result,old_crop_result=old_result,
        old_crop_transform=old_transform,old_crop_candidates=old_candidates,context_locals=locals,pools=pools,
        verification=verification,comparisons=comparisons,ground_truth_in_model_inputs=False)
    write_json(dest,result);return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['validation','test'],required=True);p.add_argument('--limit',type=int)
    args=p.parse_args();torch.set_num_threads(4)
    root=OUT/'validation_smoke' if args.stage=='validation' else OUT
    for folder in ['features','global','native','old_crop_features','context_features_0','context_features_1','pipeline','verifier','comparison']:
        (root/folder).mkdir(parents=True,exist_ok=True)
    cases=[c for c in read_jsonl(OUT/'manifest.jsonl') if c['experiment_split']==('val' if args.stage=='validation' else 'test')]
    if args.stage=='validation':cases=sorted(cases,key=lambda c:key(c['sample_id']))[:12]
    else:
        from seal_context_verifier import identity
        assert read(OUT/'inference_identity.json')==identity(),'v9 sources/models changed after seal'
    pending=[]
    for c in cases:
        path=root/'pipeline'/f'{key(c["sample_id"])}.json'
        if not path.exists():pending.append(c);continue
        saved=read(path)
        if args.stage=='validation' and any(pool and saved['comparisons'][name]['forward'].get('per_candidate_pixel_budget')!=262144
                for name,pool in saved['pools'].items()):pending.append(c)
    if args.limit:pending=pending[:args.limit]
    if pending:
        cfg=read('runs/candidate_v2/identity.json')['config'];cfg.update(max_new_tokens=64,seed=20261121,
            conditions=['prefill'],offload_folder=str((OUT/'offload').absolute()))
        cfg.update(layers=tuple(cfg['layers']),conditions=tuple(cfg['conditions']),
            max_memory={int(k) if str(k).isdigit() else k:v for k,v in cfg['max_memory'].items()})
        runtime=ContextFeatureRuntime(ExtractionConfig(**cfg));write_json(root/'runtime.json',runtime.runtime_metadata)
        models=load_predictors(primary_only=True)
        for i,c in enumerate(pending,1):
            # Strip annotations before calling any prediction function.
            public={k:c[k] for k in ['sample_id','image','image_size','image_sha256','instruction','dataset']}
            result=process(runtime,models,public,root,args.stage=='validation')
            print(json.dumps(dict(stage=args.stage,i=i,n=len(pending),sample_id=c['sample_id'],
                fixed_candidates=len(result['pools']['fixed_pool']),context_candidates=len(result['pools']['context_pool']),
                full_name=result['global_result'].get('target_name'),
                context_names=[v['result'].get('target_name') for v in result['context_locals']]),ensure_ascii=True),flush=True)
    complete=all((root/'pipeline'/f'{key(c["sample_id"])}.json').exists() for c in cases)
    write_json(root/'completion.json',dict(complete=complete,n=len(cases),ground_truth_in_model_inputs=False))


if __name__=='__main__':main()
