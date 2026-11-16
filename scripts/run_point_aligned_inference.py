"""Sealed validation smoke and resumable fresh v8 visual inference. No GT inputs."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import time

import numpy as np
from PIL import Image
import torch

from clickagent.action_benchmark import sha256,write_json
from clickagent.point_aligned_inference import proposal_crop,map_crop_point,unique_points,verify_point,VERIFIER_PROMPT
from point_aligned_data import OUT,read,archive_record
from point_aligned_predict import load_predictors,predict_record,OLD_MODELS
from run_target_token_experiment import key,read_jsonl


SOURCES=['scripts/run_point_aligned_inference.py','scripts/point_aligned_predict.py','scripts/point_aligned_data.py',
    'src/clickagent/point_aligned_inference.py','src/clickagent/point_aligned_readout.py',
    'src/clickagent/point_quality_readout.py','src/clickagent/semantic_head_experiment.py',
    'src/clickagent/model_baselines.py','scripts/run_ui_tars_baseline.py']+[
    'src/clickagent/'+n for n in ('conditioned_features.py','target_token.py','action_model.py','model_features.py','public_model.py','public_diagnostics.py')]


def identity():
    frozen=read(OUT/'frozen_readouts.json');assert frozen['complete'] and frozen['fresh_test_features_at_freeze']==0
    for p,digest in frozen['sources'].items():assert sha256(Path(p))==digest,p
    for e in frozen['models'].values():assert sha256(OUT/'models'/e['file'])==e['sha256']
    for p,digest in read(Path('runs/semantic_head_v7/frozen_readouts.json'))['sources'].items():assert sha256(Path(p))==digest,p
    cfg=read('runs/candidate_v2/identity.json')['config']
    cfg.update(max_new_tokens=64,seed=20261111,conditions=['prefill'],offload_folder=str((OUT/'offload').absolute()))
    return dict(protocol_sha256=sha256(OUT/'protocol.json'),frozen_sha256=sha256(OUT/'frozen_readouts.json'),
        manifest_sha256=sha256(OUT/'manifest.jsonl'),sources={p:sha256(Path(p)) for p in SOURCES},
        old_models={p:sha256(Path(p)) for _,p in OLD_MODELS.values()},
        qwen_weight_identity_sha256=sha256(Path('data/conditioned_v3/qwen_weight_identity.json')),
        ui_tars_verified_identity_sha256=sha256(Path('models/UI-TARS-1.5-7B/clickagent_download_verified.json')),
        libraries={p:importlib.metadata.version(p) for p in ('torch','transformers','accelerate','numpy','Pillow')},
        config=cfg,verifier_prompt=VERIFIER_PROMPT,ground_truth_used_in_predictors=False,
        inference_protocol=read(OUT/'protocol.json')['inference'])


def save_features(path,arrays,meta):
    path.parent.mkdir(exist_ok=True,parents=True);temporary=path.with_suffix('.tmp')
    with temporary.open('wb') as stream:np.savez_compressed(stream,metadata=np.asarray(json.dumps(meta,ensure_ascii=False)),**arrays)
    temporary.replace(path)


def extract(runtime,image,instruction,path,case,**extra):
    instruction_sha=hashlib.sha256(instruction.encode('utf-8')).hexdigest()
    if not path.exists():
        arrays,meta=runtime.infer_spatial_features(image,instruction)
        assert meta['all_backbone_parameters_frozen'] and meta['own_token_forward'] and not meta['ground_truth_in_model_input']
        meta.update(sample_id=case['sample_id'],image_sha256=case['image_sha256'],instruction_sha256=instruction_sha,**extra)
        save_features(path,arrays,meta)
    record=archive_record(path,case,labels=False);meta=record['metadata']
    assert meta['sample_id']==case['sample_id'] and meta['image_sha256']==case['image_sha256']
    assert meta['instruction_sha256']==instruction_sha
    for name,value in extra.items():assert meta[name]==value
    return record


def native(runtime,image,instruction,path,transform=None):
    if not path.exists():
        from clickagent.public_diagnostics import parse_offline_lax_click
        result=runtime.infer_text_action_public(image,instruction,max_new_tokens=32)
        complete=result['terminated_by_eos'] or result['generated_text'].endswith(('\n','\r'))
        parsed=parse_offline_lax_click(result['generated_text']) if complete else None
        result['completed_lax_point']=[round(v*(size-1)/1000) for v,size in zip(parsed,image.size)] if parsed else None
        if transform is not None:result['original_point']=map_crop_point(result['completed_lax_point'],transform)
        write_json(path,result)
    return read(path)


def process_case(runtime,models,case,root):
    sid=key(case['sample_id']);primary='joint_connected_s1'
    assert sha256(Path(case['image']))==case['image_sha256']
    with Image.open(case['image']) as opened:image=opened.convert('RGB')
    record=extract(runtime,image,case['instruction'],root/'features'/f'{sid}.npz',case)
    gp=root/'global'/f'{sid}.json'
    if not gp.exists():write_json(gp,dict(sample_id=case['sample_id'],predictions=predict_record(record,models)))
    global_predictions=read(gp)['predictions'];global_result=global_predictions[primary]
    native(runtime,image,case['instruction'],root/'native'/f'{sid}.json')
    pipeline_path=root/'pipeline'/f'{sid}.json'
    if pipeline_path.exists():return read(pipeline_path)
    started=time.perf_counter();crop_result=None;crop_candidates=[];crop_native=None;transform=None
    if global_result['candidates']:
        top=global_result['candidates'][0]
        crop,transform=proposal_crop(image,top['parent_index'],record['metadata']['grid_shape'])
        crop_case=dict(sample_id=case['sample_id'],image_size=list(crop.size),image_sha256=case['image_sha256'])
        crop_record=extract(runtime,crop,case['instruction'],root/'crop_features'/f'{sid}.npz',crop_case,transform=transform)
        crop_result=predict_record(crop_record,models,only=[primary])[primary]
        for c in crop_result['candidates']:
            crop_candidates.append(dict(c,local_point=c['point'],point=map_crop_point(c['point'],transform),source='crop'))
        crop_native=native(runtime,crop,case['instruction'],root/'crop_native'/f'{sid}.json',transform)
    global_candidates=[dict(c,source='global') for c in global_result['candidates']]
    global_pool=unique_points(global_candidates[:4]);combined_pool=unique_points(global_candidates[:2],crop_candidates[:2],limit=4)
    verification={};union=unique_points(global_pool,combined_pool)
    for c in union:
        x,y=c['point'];vp=root/'verifier'/f'{sid}_{x}_{y}.json'
        if not vp.exists():write_json(vp,verify_point(runtime,image,case['instruction'],c['point']))
        verification[f'{x},{y}']=read(vp)
    def rank(pool):
        if not pool:return None
        return max(pool,key=lambda c:verification[','.join(map(str,c['point']))]['score'])
    global_verified=rank(global_pool);combined_verified=rank(combined_pool)
    result=dict(sample_id=case['sample_id'],transform=transform,crop_result=crop_result,
        crop_point=crop_candidates[0]['point'] if crop_candidates else None,
        crop_native_point=crop_native['original_point'] if crop_native else None,
        global_pool=global_pool,combined_pool=combined_pool,verification=verification,
        global_verified_point=global_verified['point'] if global_verified else None,
        combined_verified_point=combined_verified['point'] if combined_verified else None,
        verifier_unique_passes=len(union),crop_pipeline_elapsed_seconds=time.perf_counter()-started,
        ground_truth_in_model_inputs=False)
    write_json(pipeline_path,result);return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['smoke','seal','test'],required=True);parser.add_argument('--limit',type=int)
    args=parser.parse_args();torch.set_num_threads(4)
    current=identity();path=OUT/'inference_identity.json'
    if args.stage=='seal':
        if path.exists():
            assert read(path)==current;print('INFERENCE ALREADY SEALED',flush=True);return
        smoke=read(OUT/'validation_smoke'/'completion.json');assert smoke['complete'] and smoke['identity']==current
        assert not list((OUT/'features').glob('*.npz'))
        write_json(path,current);snapshot=OUT/'inference_sources';snapshot.mkdir(exist_ok=True)
        for p in SOURCES:shutil.copyfile(p,snapshot/Path(p).name)
        print('INFERENCE SEALED AFTER VALIDATION SMOKE',flush=True);return
    rows=read_jsonl(OUT/'manifest.jsonl')
    if args.stage=='test':
        assert read(path)==current,'Inference source or identity changed after sealing'
        cases=[c for c in rows if c['experiment_split']=='test'];root=OUT
    else:
        root=OUT/'validation_smoke';root.mkdir(exist_ok=True)
        cases=sorted([c for c in rows if c['experiment_split']=='val'],key=lambda c:key(c['sample_id']))[:12]
        smoke_identity=root/'identity.json'
        if smoke_identity.exists():assert read(smoke_identity)==current,'Use a new smoke namespace after code changes'
        write_json(smoke_identity,current)
    for folder in ('features','global','native','crop_features','crop_native','pipeline','verifier'):(root/folder).mkdir(parents=True,exist_ok=True)
    pending=[c for c in cases if not (root/'pipeline'/f'{key(c["sample_id"])}.json').exists()]
    if args.limit:pending=pending[:args.limit]
    if pending:
        from clickagent.model_features import ExtractionConfig
        from clickagent.conditioned_features import ConditionedFeatureRuntime
        cfg=dict(current['config']);cfg.update(layers=tuple(cfg['layers']),conditions=tuple(cfg['conditions']),
            max_memory={int(k) if str(k).isdigit() else k:v for k,v in cfg['max_memory'].items()})
        runtime=ConditionedFeatureRuntime(ExtractionConfig(**cfg));write_json(root/'runtime.json',runtime.runtime_metadata)
        models=load_predictors()
        for i,c in enumerate(pending,1):
            result=process_case(runtime,models,c,root)
            print(json.dumps(dict(stage=args.stage,i=i,n=len(pending),sample_id=c['sample_id'],
                crop_point=result['crop_point'],combined_verified=result['combined_verified_point'],verifier_passes=result['verifier_unique_passes'])),flush=True)
    complete=all((root/'pipeline'/f'{key(c["sample_id"])}.json').exists() for c in cases)
    write_json(root/'completion.json',dict(complete=complete,n=len(cases),identity=current))
    print('COMPLETE '+str(complete),flush=True)


if __name__=='__main__':main()
