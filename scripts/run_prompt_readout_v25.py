"""V25 prompt/readout/composition pilot, fixed views and matched small adaptation."""
import os
os.environ.setdefault('PYTHONDONTWRITEBYTECODE','1')
os.environ.setdefault('HF_HUB_OFFLINE','1')
os.environ.setdefault('TRANSFORMERS_OFFLINE','1')
os.environ.setdefault('OMP_NUM_THREADS','4')
os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse
from collections import Counter,defaultdict
from dataclasses import asdict
from datetime import datetime,timezone
import gc
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
import traceback
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import numpy as np
import torch
from PIL import Image,ImageDraw
from clickagent.prompt_readout_v25 import (PROMPTS,ARMS,ALL_ARMS,PromptReadoutRuntime,select_arrays,
    compose_arrays,normalized_to_pixel,map_view_point,coordinate_consensus)
from clickagent.crossfit_proposer import load_crossfit_proposer,predict_raw,state_sha256
from clickagent.evidence_retention import RetentionModel,STATE_DIM,build_base,state_features,retention_loss
from clickagent.multiregion_proposals import build_c1_c3_pools
from clickagent.model_features import ExtractionConfig
from run_visual_feedback_v23 import read,write,sha,hit,bootstrap
from clickagent.visual_feedback_v23 import digest
from run_evidence_retention_v14 import tensors

OUT=ROOT/'runs/prompt_readout_v25'
PRIOR=ROOT/'runs/visual_feedback_v23'
V21=ROOT/'runs/mvp_repair_v21'
MAIN=ROOT/'runs/quality_readout_v15_3k'
V12=ROOT/'runs/crossfit_candidates_v12'
NAMES=[f'{family}_s{s}' for family in ['original3k','full'] for s in [1,2,3]]
SOURCES=[Path(__file__),ROOT/'src/clickagent/prompt_readout_v25.py',ROOT/'tests/test_prompt_readout_v25.py']
PUBLIC=('sample_id','dataset','image','image_size','image_sha256','instruction')
PAIRS={'context_prompt':('context_name','name'),'coordinate_task':('coordinate_mean','name'),
    'within_coordinate_stage':('coordinate_mean','coordinate_name'),
    'whole_phrase':('framed_phrase','context_name'),'product_vs_entity':('role_product','entity_mean'),
    'product_vs_sum':('role_product','role_sum'),'location_vs_entity':('location_gate','entity_mean')}
TRAIN_STEPS=64


def now():return datetime.now(timezone.utc).isoformat()
def key(s):return hashlib.sha256(s.encode()).hexdigest()[:24]
def source_identity():return {str(p):sha(p) for p in SOURCES}


def status(phase,**fields):
    row=dict(phase=phase,complete=phase=='complete',updated_at_utc=now(),pid=os.getpid(),**fields)
    write(OUT/'status.json',row);print(json.dumps(row,ensure_ascii=False),flush=True)


def npz(path,**arrays):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp')
    with tmp.open('wb') as f:np.savez_compressed(f,**arrays)
    os.replace(tmp,path)


def load_npz(path):
    with np.load(path,allow_pickle=False) as z:
        meta=json.loads(str(z['metadata']));arrays={k:z[k].copy() for k in z.files if k!='metadata'}
    return arrays,meta


def select_inputs():
    indices=read(ROOT/'runs/bottleneck_v19/data/indices.json')
    selected=[]
    groups=defaultdict(lambda:defaultdict(list))
    for e in indices['train']:groups[e['source']][e['group']].append(e)
    for source,gg in sorted(groups.items()):
        for group in sorted(gg,key=lambda g:digest([g,'v25_train_rgb']))[:8]:
            selected.append(('train',min(gg[group],key=lambda e:digest([e['sample_id'],'v25_query']))))
    assert len(selected)==56
    prior_cases=read(PRIOR/'data/public.json');prior_indices=read(PRIOR/'data/indices.json')
    for cohort,count in [('new_development',8),('old_val',3)]:
        for source in sorted({c['source'] for c in prior_cases if c['cohort']==cohort}):
            eligible=[c for c in prior_cases if c['source']==source and c['cohort']==cohort]
            for c in sorted(eligible,key=lambda c:digest([c['group'],'v25_dev_rgb']))[:count]:
                selected.append((cohort,prior_indices[c['sample_id']]))
    assert len(selected)==92
    manifests=[ROOT/'runs/multiregion_v13/data/train.jsonl',ROOT/'runs/multiregion_v13/data/val.jsonl',
               MAIN/'data/new_train.jsonl',MAIN/'data/new_reserve.jsonl']
    lookup={}
    for path in manifests:
        for line in path.read_text(encoding='utf-8-sig').splitlines():
            c=json.loads(line);lookup[c['sample_id']]=c
    cases=[];labels={};entries={}
    for cohort,e in selected:
        c=lookup[e['sample_id']];p=read(e['pipeline_path'])
        public={k:c[k] for k in PUBLIC}
        public.update(cohort=cohort,source=e['source'],group=e['group'],source_group=e['source_group'])
        public['views']=[dict(feature_path=p['global_feature_path'],feature_sha256=p['global_feature_sha256'],transform=None)]
        public['views'] += [{k:v[k] for k in ['feature_path','feature_sha256','transform']} for v in p['local_views']]
        cases.append(public);labels[c['sample_id']]=c['target_geometry'];entries[c['sample_id']]=e
    cases.sort(key=lambda c:(c['cohort']!='train',digest([c['sample_id'],'v25_case_order'])))
    assert len({c['group'] for c in cases})==92
    return cases,labels,entries,manifests


def task_for(c,view_index,prompt):
    view=c['views'][view_index]
    identity=dict(sample_id=c['sample_id'],image_sha256=c['image_sha256'],instruction=c['instruction'],
        transform=view['transform'],prompt=prompt,prompt_sha256=digest(PROMPTS[prompt]),max_new_tokens=96,
        contract_sha256=sha(V12/'runtime_identity.json'),schema='v25_full_own_forward_trajectory')
    return dict(sample_id=c['sample_id'],view_index=view_index,prompt=prompt,task_id=digest(identity)[:24],input_identity=digest(identity))


def verify():
    p=read(OUT/'protocol.json');assert p['source_sha256']==source_identity(),'V25 source drift'
    for path,h in p['sealed_sha256'].items():assert sha(path)==h,path
    for path,r in p['backbone_identity'].items():
        st=Path(path).stat();assert st.st_size==r['bytes'] and st.st_mtime_ns==r['mtime_ns'],path
    return p


def prepare():
    if (OUT/'protocol.json').exists():return verify()
    test=read(OUT/'unit_tests.json');assert test['exit_code']==0 and test['source_sha256']==source_identity()
    smoke=read(OUT/'smoke/receipt.json');assert smoke['complete'] and smoke['source_sha256']==source_identity()
    assert smoke['original_target_replay_passed']
    cases,labels,entries,manifests=select_inputs();provenance={};tasks=[]
    for i,c in enumerate(cases,1):
        e=entries[c['sample_id']];assert sha(c['image'])==c['image_sha256']
        for field,hfield in [('path','sha256'),('state_path','state_sha256'),('mask_path','mask_sha256'),('pipeline_path','pipeline_sha256')]:
            assert sha(e[field])==e[hfield],e[field];provenance[e[field]]=e[hfield]
        for vi,v in enumerate(c['views']):
            assert sha(v['feature_path'])==v['feature_sha256'];provenance[v['feature_path']]=v['feature_sha256']
            for prompt in ['context','coordinate']:tasks.append(task_for(c,vi,prompt))
        if i%10==0:status('sealing_fixed_input_views',completed=i,total=len(cases))
    for name,value in [('public',cases),('labels',labels),('indices',entries),('tasks',tasks),('provenance',provenance)]:
        write(OUT/f'data/{name}.json',value)
    prior=read(PRIOR/'protocol.json')
    dependencies=['conditioned_features','target_token','action_model','public_model','model_features',
        'crossfit_proposer','candidate_ranker','semantic_head_ranker','attention_field_ranker','point_aligned_readout',
        'point_aligned_inference','evidence_retention','multiregion_proposals','public_evaluation']
    sealed=SOURCES+manifests+[OUT/f'data/{n}.json' for n in ['public','labels','indices','tasks','provenance']]
    sealed += [ROOT/f'src/clickagent/{n}.py' for n in dependencies]
    sealed += [ROOT/'scripts/run_evidence_retention_v14.py',OUT/'unit_tests.json',OUT/'smoke/receipt.json',
        V12/'runtime_identity.json',V12/'proposers/full.pt',PRIOR/'protocol.json',PRIOR/'predictions.json',
        V21/'protocol.json',V21/'results.json']+[Path(v['path']) for v in prior['models'].values()]
    spec=dict(registered_at_utc=now(),source_sha256=source_identity(),sealed_sha256={str(p):sha(p) for p in sealed},
        backbone_identity=prior['backbone_identity'],prompts=PROMPTS,model_names=NAMES,models=prior['models'],
        samples='56 original-training RGBs(8/source7), 36 historically reused development RGBs(new24=8/source3, old12=3/source4), one query/RGB. Deterministic SHA sampling independent of outcomes. Zero train/development RGB overlap. No fresh280.',
        fixed_views='Exactly original name-prompt global+up-to3 crop boxes and crop sizes for every prompt/readout. Frozen original611919-parameter proposer and complete candidate rule. New prompts cannot replan views. This tests prompt/readout under fixed original observation, not a new end-to-end crop strategy.',
        prompt_groups='A original2-line name prompt, exact original cached features; B grounded container/location/entity frame; C target-name then coordinates. B/C capture the full actual generated trajectory before choosing role/token stages. Complete spans only; actual own-forward indices, never retokenized IDs.',
        arms=list(ALL_ARMS),adapted_arms=list(ARMS),
        composition='Framed B entity-only and all-content-token controls. Entity_mean, role_sum, role_product, location_gate all use identical mean entity hidden query and entity image-attention mass. Per-head spatial maps normalized on same native grid, modifiers weight0.5. Sum is weighted arithmetic mixture; product sums weighted log maps(eps1e-8) and normalizes, no independence/probability claim. Location_gate uses predicted current-image9-bin location and Gaussian sigma0.35; unknown keeps entity map. These operations precede peak extraction and pruning. No arbitrary relative-anchor relation execution is implemented.',
        parsing='Complete ground frame with exactly one container/location/target field; NONE/unknown modifier omitted. Complete POINT line integers0..1000, optional balanced parentheses/brackets and comma/space separators. Malformed/capped unfinished spans remain absent. No GT-derived repair or fallback to another prompt.',
        generation='Frozen same8B FP16,layers15/28/36,262144 pixels/view, greedy; original A cached64-token cap, new B/C96-token cap to accommodate framing. Actual token counts/time recorded; not equal-output-compute claim. Synthetic A replay uses original64 cap.',
        adaptation=dict(architecture='Original22050-parameter RetentionModel',initialization='Each of six original3k/full checkpoints cloned identically acrossarms',
            preselected_train_rgb=56,steps=TRAIN_STEPS,batch_size=4,lr=2e-5,weight_decay=.02,grad_clip=1.,
            normalization='Original checkpoint mean/std frozen identically acrossarms',
            data='Common nonempty56-training subset across all9 adapted arms, selected by feature availability only. Same batch sample-ID sequence per seed acrossallarms. Retain all development failures. No development checkpoint/prompt/seed selection.',
            stages='Extract all training features, train all54 small adapters and seal their hashes BEFORE any new development B/C feature extraction. Original weights and backbone unchanged. If common subset empty, adapters copy unchanged originals with0 steps and explicit insufficient-data status.'),
        primary='Adapted families3k/full x7 registered contrasts='+str(PAIRS)+'. New24 only,5000 source-stratified RGB bootstrap over3 fixed-fit means,99.642857% Bonferroni14 intervals. Tiny development pilot, no independent confirmation.',
        secondary='Frozen6-head prompt/readout comparisons, old12/all36, sameC trajectory name vs coordinate/separator, nativeC global/multiview coordinates, cachedofficialMVP on same screenshots, full/top3/top9 candidate coverage, raw mean-attention peak diagnostics, complete phrase vs entity, tokens/time/parse failures.',
        advancement='Development signal only: adapted B-name or C-coordinate minus adapted A >=3pp with corrected lower>0 and old12 delta>=-5pp. No automatic adoption; no claims of outperforming MVP without independent test and declared compute.',
        actual_new_vlm_trajectories=len(tasks),ground_truth='Labels stored separately, never supplied to trajectory generation, view planning, readout selection, fusion or proposal construction. Training labels used only after label-free candidate features exist.',
        user_authorization='要不试试吧；并入用户提出的框定完整指代表达、组合各短语视觉证据。',
        outputs='Python logs/JSON/TXT/NPZ/PT/staticPNG only; no HTML, no actual OS actions.')
    write(OUT/'registry_before.json',read(ROOT/'CURRENT_RESEARCH.json'));write(OUT/'protocol.json',spec)
    for path in SOURCES:
        dest=OUT/'source_archive'/path.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,dest)
    reg=read(ROOT/'CURRENT_RESEARCH.json');reg['active_experiment']=dict(run_id=OUT.name,status='registered',phase='prepared',
        protocol_path=str(OUT/'protocol.json'),status_path=str(OUT/'status.json'),training_rgb=56,development_rgb=36,
        actual_vlm_trajectories=len(tasks),mainline_unchanged=True)
    write(ROOT/'CURRENT_RESEARCH.json',reg)
    write(ROOT/'research/decisions/20261109_prompt_readout_v25.json',dict(user_authorization=spec['user_authorization'],
        registered_at_utc=now(),protocol_sha256=sha(OUT/'protocol.json'),mainline_unchanged=True))
    status('registered',training_rgb=56,development_rgb=36,vlm_trajectories=len(tasks))
    return spec


def load_runtime():
    from clickagent.advantage_runtimes import configure_paths
    configure_paths('coordinate');torch.set_num_threads(4)
    cfg=dict(read(V12/'runtime_identity.json')['runtime_contract']);cfg['layers']=tuple(cfg['layers']);cfg['conditions']=tuple(cfg['conditions'])
    cfg['max_memory']={int(k) if str(k).isdigit() else k:v for k,v in cfg['max_memory'].items()}
    cfg['offload_folder']=str(OUT/'offload')
    return PromptReadoutRuntime(ExtractionConfig(**cfg))


def opened_view(c,view):
    assert sha(c['image'])==c['image_sha256']
    with Image.open(c['image']) as raw:image=raw.convert('RGB')
    transform=view['transform']
    if transform is not None:
        crop=image.crop(transform['box']).resize(tuple(transform['crop_size']),Image.Resampling.LANCZOS)
        image.close();image=crop
    return image


def smoke():
    status('synthetic_and_original_replay_loading');runtime=load_runtime();receipts=[]
    image=Image.new('RGB',(1000,700),'white');d=ImageDraw.Draw(image)
    d.rectangle((80,100,300,260),fill=(220,20,20));d.rectangle((650,430,900,600),fill=(20,70,220))
    for prompt in ['context','coordinate']:
        arrays,meta=runtime.infer_trajectory(image,'Click the blue rectangle.',prompt,96)
        assert len(arrays['all_attention'])==len(meta['generated_token_ids'])
        assert meta['own_token_forward'] and meta['all_backbone_parameters_frozen']
        if prompt=='context':assert meta['ground_complete'] and meta['target_available'],meta['generated_text']
        else:assert meta['coordinate_available'] and meta['target_available'],meta['generated_text']
        npz(OUT/f'smoke/{prompt}.npz',metadata=np.asarray(json.dumps(meta,ensure_ascii=False)),**arrays)
        receipts.append(dict(prompt=prompt,text=meta['generated_text'],stages=meta['readout_stages'],seconds=meta['elapsed_seconds']))
    image.close();cases,_,_,_=select_inputs();training=[c for c in cases if c['cohort']=='train']
    c=training[0];replays=[]
    for vi in range(min(2,len(c['views']))):
        v=c['views'][vi];im=opened_view(c,v);a,m=runtime.infer_trajectory(im,c['instruction'],'name',64);im.close()
        selected,meta=select_arrays(a,m,'target');old,om=load_npz(v['feature_path'])
        assert m['generated_token_ids']==om['generated_token_ids']
        errors={}
        for name in ['target_attention','target_hidden','image_hidden','vision_hidden']:
            assert old[name].shape==selected[name].shape
            assert np.allclose(old[name],selected[name],atol=2e-4,rtol=2e-4),name
            errors[name]=float(np.max(np.abs(old[name].astype(np.float32)-selected[name].astype(np.float32)))) if old[name].size else 0.
        replays.append(dict(sample_id=c['sample_id'],view_index=vi,max_errors=errors))
    write(OUT/'smoke/receipt.json',dict(complete=True,at_utc=now(),source_sha256=source_identity(),
        synthetic_outputs=receipts,original_target_replay_passed=True,replays=replays,runtime_metadata=runtime.runtime_metadata))
    status('synthetic_and_original_replay_complete',replay_views=len(replays))


def readout_data(c,vi,arm):
    if arm=='name':
        a,m=load_npz(c['views'][vi]['feature_path']);return {k:a[k] for k in ['target_attention','target_hidden','image_hidden','vision_hidden']},m
    prompt='coordinate' if arm.startswith('coordinate_') else 'context'
    task=task_for(c,vi,prompt);a,m=load_npz(OUT/f'raw/{task["task_id"]}.npz')
    assert m['input_identity']==task['input_identity']
    if arm in ('entity_mean','role_sum','role_product','location_gate'):return compose_arrays(a,m,arm)
    stage={'context_name':'target','coordinate_name':'target','coordinate_mean':'coordinate',
           'coordinate_separator':'separator','framed_phrase':'phrase'}[arm]
    return select_arrays(a,m,stage)


def feature_path(c,arm):return OUT/f'features/{c["cohort"]}/{arm}/{key(c["sample_id"])}.npz'


def build_features(c,arm,proposer):
    dest=feature_path(c,arm)
    if dest.exists():
        _,meta=load_npz(dest)
        assert meta['protocol_sha256']==sha(OUT/'protocol.json') and meta['source_sha256']==source_identity()
        assert meta['sample_id']==c['sample_id'] and meta['arm']==arm
        return
    arrays=[];metas=[];decoded=[];raw_peaks=[]
    for vi,v in enumerate(c['views']):
        a,m=readout_data(c,vi,arm);arrays.append(a);metas.append(m)
        size=c['image_size'] if v['transform'] is None else v['transform']['crop_size']
        result=predict_raw(proposer,a,m,size,topk=64);decoded.append(result)
        if len(a['target_attention']):
            rel=a['target_attention']/np.maximum(a['target_attention'].sum(-1,keepdims=True),1e-12)
            index=int(rel.mean((0,1)).argmax());gh,gw=m['grid_shape'];row,col=divmod(index,gw)
            point=[min(size[0]-1,round((col+.5)*size[0]/gw)),min(size[1]-1,round((row+.5)*size[1]/gh))]
            raw_peaks.append(map_view_point(point,v['transform']))
        else:raw_peaks.append(None)
    locals=[dict(result=decoded[i],transform=v['transform']) for i,v in enumerate(c['views']) if i>0]
    pipe=dict(global_result=decoded[0],local_views=locals,pools=build_c1_c3_pools(decoded[0],locals))
    base=build_base(pipe,c['image_size']);points=base['points'];states=[];valid=[]
    for a,m,v in zip(arrays,metas,c['views']):
        f,mask=state_features(a,m,points,c['image_size'],v['transform']);states.append(f);valid.append(mask)
    tmax=max(1,*(x.shape[1] for x in states));state=np.zeros((len(points),4,tmax,STATE_DIM),np.float16);mask=np.zeros((len(points),4,tmax),bool)
    for i,(f,v) in enumerate(zip(states,valid)):
        t=f.shape[1];state[:,i,:t]=f;mask[:,i,:t]=v[:,None]
    meta=dict(sample_id=c['sample_id'],arm=arm,source_sha256=source_identity(),protocol_sha256=sha(OUT/'protocol.json'),
        views=len(c['views']),readout_available=[bool(m['target_available']) for m in metas],
        target_names=[m.get('target_text') for m in metas],raw_peak_points=raw_peaks,
        ground_truth_in_model_inputs=False,candidate_rule='unchanged own proposer complete pool, fixed original views')
    npz(dest,metadata=np.asarray(json.dumps(meta,ensure_ascii=False)),state=state,mask=mask,**base)


def load_record(c,arm,geometry=None):
    a,m=load_npz(feature_path(c,arm));r=dict(sample_id=c['sample_id'],n=len(a['points']),**a)
    if geometry is not None:r['labels']=np.asarray([hit(p.tolist(),c,geometry) for p in a['points']],np.float32)
    return r


def extract_role(role,runtime,proposer):
    cases=[c for c in read(OUT/'data/public.json') if (c['cohort']=='train')==(role=='train')]
    tasks=[task_for(c,vi,p) for c in cases for vi in range(len(c['views'])) for p in ['context','coordinate']]
    if role!='train':assert read(OUT/'models/frozen.json')['complete'],'Freeze adapters before development extraction'
    protocol_sha=sha(OUT/'protocol.json');done=0;started=time.perf_counter()
    for c in cases:
        for vi,v in enumerate(c['views']):
            for prompt in ['context','coordinate']:
                task=task_for(c,vi,prompt);path=OUT/f'raw/{task["task_id"]}.npz'
                if path.exists():
                    _,meta=load_npz(path);assert meta['input_identity']==task['input_identity'] and meta['protocol_sha256']==protocol_sha
                else:
                    im=opened_view(c,v);capture_started=now()
                    a,m=runtime.infer_trajectory(im,c['instruction'],prompt,96);im.close()
                    m.update(input_identity=task['input_identity'],task=task,protocol_sha256=protocol_sha,
                             sample_id=c['sample_id'],image_sha256=c['image_sha256'],transform=v['transform'],
                             capture_started_at_utc=capture_started,capture_completed_at_utc=now())
                    npz(path,metadata=np.asarray(json.dumps(m,ensure_ascii=False)),**a)
                done+=1;status('extracting_'+role,completed=done,total=len(tasks),elapsed_seconds=time.perf_counter()-started)
        for arm in ALL_ARMS:build_features(c,arm,proposer)
        if role=='train':status('training_case_features_complete',completed_cases=cases.index(c)+1,total_cases=len(cases),trajectory_completed=done,trajectory_total=len(tasks))
    write(OUT/f'{role}_features_complete.json',dict(complete=True,at_utc=now(),cases=len(cases),trajectories=len(tasks),protocol_sha256=protocol_sha))


def train_adapters():
    assert read(OUT/'train_features_complete.json')['complete'];spec=verify()
    if (OUT/'models/frozen.json').exists():
        frozen=read(OUT/'models/frozen.json')
        assert frozen['complete'] and frozen['protocol_sha256']==sha(OUT/'protocol.json')
        assert len(frozen['models'])==len(ARMS)*len(NAMES)
        for row in frozen['models'].values():assert sha(row['checkpoint_path'])==row['checkpoint_sha256']
        return frozen
    cases=[c for c in read(OUT/'data/public.json') if c['cohort']=='train'];labels=read(OUT/'data/labels.json')
    common=[c for c in cases if all(load_record(c,a)['n']>0 for a in ARMS)]
    write(OUT/'common_training.json',dict(preselected=56,n=len(common),sample_ids=[c['sample_id'] for c in common],
        availability_only=True,all_arms_same_samples=True))
    models={};torch.set_num_threads(4)
    for arm in ARMS:
        records=[load_record(c,arm,labels[c['sample_id']]) for c in common]
        for name in NAMES:
            tag=f'{arm}__{name}';dest=OUT/f'models/{tag}.pt';receipt=OUT/f'models/{tag}.json'
            if receipt.exists():
                row=read(receipt);assert sha(dest)==row['checkpoint_sha256'];models[tag]=row;continue
            original=spec['models'][name];saved=torch.load(original['path'],map_location='cpu',weights_only=True)
            model=RetentionModel();model.load_state_dict(saved['state_dict']);assert sum(p.numel() for p in model.parameters())==22050
            initial=state_sha256(model);mean=saved['mean'].numpy();std=saved['std'].numpy();seed=20262510+int(name[-1])
            torch.manual_seed(seed);optimizer=torch.optim.AdamW(model.parameters(),lr=2e-5,weight_decay=.02)
            batches=[];losses=[];started=time.perf_counter();steps=TRAIN_STEPS if records else 0
            rng=np.random.default_rng(seed);order=[];cursor=0;model.train()
            for step in range(steps):
                chosen=[]
                while len(chosen)<4:
                    if cursor==len(order):order=rng.permutation(len(records)).tolist();cursor=0
                    chosen.append(order[cursor]);cursor+=1
                batch=[records[i] for i in chosen];batches.append([r['sample_id'] for r in batch])
                inputs,lengths=tensors(batch,mean,std,'aligned','cpu');optimizer.zero_grad(set_to_none=True)
                scores=model(**inputs);loss=torch.stack([retention_loss(s,torch.as_tensor(r['labels']),model.null) for s,r in zip(scores.split(lengths),batch)]).mean()
                assert torch.isfinite(loss)
                loss.backward();assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();losses.append(float(loss.detach()))
                if (step+1)%16==0:status('adapting_readouts',model=tag,step=step+1,total_steps=steps,common_training_rgb=len(common))
            model.eval();dest.parent.mkdir(parents=True,exist_ok=True)
            torch.save(dict(state_dict=model.state_dict(),mean=saved['mean'],std=saved['std'],variant='aligned',seed=seed),dest)
            row=dict(complete=True,name=tag,arm=arm,original_model=name,checkpoint_path=str(dest),checkpoint_sha256=sha(dest),
                initial_state_sha256=initial,original_checkpoint_sha256=original['sha256'],steps=steps,training_rgb=len(common),
                batch_sequence_sha256=digest(batches),parameter_count=22050,mean_loss=float(np.mean(losses)) if losses else None,
                elapsed_seconds=time.perf_counter()-started,normalization_sha256=digest([mean.tolist(),std.tolist()]),
                protocol_sha256=sha(OUT/'protocol.json'),development_used=False)
            write(receipt,row);models[tag]=row
            del model,optimizer;gc.collect()
        del records;gc.collect()
    for name in NAMES:
        rows=[models[f'{arm}__{name}'] for arm in ARMS]
        assert len({r['initial_state_sha256'] for r in rows})==1
        assert len({r['batch_sequence_sha256'] for r in rows})==1
        assert len({r['normalization_sha256'] for r in rows})==1
    write(OUT/'models/frozen.json',dict(complete=True,frozen_at_utc=now(),models=models,
        development_new_prompt_outputs_not_yet_generated=True,common_training_rgb=len(common),protocol_sha256=sha(OUT/'protocol.json')))
    status('adapters_frozen',models=len(models),common_training_rgb=len(common))


def worker():
    verify();status('loading_frozen8B_and_proposer');runtime=load_runtime()
    write(OUT/'runtime_metadata.json',runtime.runtime_metadata)
    proposer=load_crossfit_proposer(V12/'proposers/full.pt','cpu')[0].eval().requires_grad_(False)
    extract_role('train',runtime,proposer);train_adapters();extract_role('development',runtime,proposer)
    status('all_features_and_training_complete')


def evaluate():
    spec=verify();assert read(OUT/'development_features_complete.json')['complete']
    cases=[c for c in read(OUT/'data/public.json') if c['cohort']!='train'];labels=read(OUT/'data/labels.json')
    frozen=read(OUT/'models/frozen.json');rows=[];costs={};native={};prior_rows=read(PRIOR/'predictions.json')['rows']
    prior={(r['sample_id'],r['model']):r for r in prior_rows if r['arm']=='baseline'}
    torch.set_num_threads(4);models={}
    for name in NAMES:
        saved=torch.load(spec['models'][name]['path'],map_location='cpu',weights_only=True);model=RetentionModel().eval();model.load_state_dict(saved['state_dict']);models['frozen',name,None]=(model,saved)
        for arm in ARMS:
            info=frozen['models'][arm+'__'+name];assert sha(info['checkpoint_path'])==info['checkpoint_sha256']
            saved=torch.load(info['checkpoint_path'],map_location='cpu',weights_only=True);model=RetentionModel().eval();model.load_state_dict(saved['state_dict']);models['adapted',name,arm]=(model,saved)
    parity=0
    for c in cases:
        sid=c['sample_id'];points=[];costs[sid]={}
        for prompt in ['context','coordinate']:
            metas=[load_npz(OUT/f'raw/{task_for(c,i,prompt)["task_id"]}.npz')[1] for i in range(len(c['views']))]
            costs[sid][prompt]=dict(seconds=sum(m['elapsed_seconds'] for m in metas),tokens=sum(len(m['generated_token_ids']) for m in metas),
                views=len(metas),caps=sum(m['token_cap_reached'] for m in metas),available=sum(m['target_available'] if prompt=='context' else m['coordinate_available'] for m in metas))
            if prompt=='coordinate':
                points=[map_view_point(normalized_to_pixel(m['normalized_point'],m['image_size']),v['transform']) for m,v in zip(metas,c['views'])]
        native[sid]=dict(global_point=points[0],consensus_point=coordinate_consensus(points),view_points=points)
        key21=key(sid);native[sid]['MVP']=read(V21/f'mvp/{key21}.json')['output']['point']
        for arm in ALL_ARMS:
            r=load_record(c,arm);oracle=any(hit(p.tolist(),c,labels[sid]) for p in r['points'])
            _,fm=load_npz(feature_path(c,arm))
            for regime in ['frozen','adapted']:
                if regime=='adapted' and arm not in ARMS:continue
                for name in NAMES:
                    model,saved=models[regime,name,None if regime=='frozen' else arm]
                    if r['n']:
                        inputs,_=tensors([r],saved['mean'].numpy(),saved['std'].numpy(),'aligned','cpu')
                        with torch.inference_mode():scores=model(**inputs).numpy()
                        point=r['points'][int(scores.argmax())].tolist()
                        order=np.argsort(-scores,kind='stable');top3=r['points'][order[:3]];top9=r['points'][order[:9]]
                    else:point=None;top3=top9=[]
                    correct=hit(point,c,labels[sid])
                    if regime=='frozen' and arm=='name':assert correct==prior[sid,name]['hit'] and point==prior[sid,name]['point'];parity+=1
                    rows.append(dict(sample_id=sid,regime=regime,model=name,arm=arm,point=point,hit=correct,
                        candidate_count=r['n'],oracle=oracle,top3_oracle=any(hit(p.tolist(),c,labels[sid]) for p in top3),
                        top9_oracle=any(hit(p.tolist(),c,labels[sid]) for p in top9),raw_peak_any_hit=any(hit(p,c,labels[sid]) for p in fm['raw_peak_points']),
                        readout_available_views=fm['readout_available']))
        status('evaluating_fixed_predictions',completed=cases.index(c)+1,total=len(cases))
    write(OUT/'predictions.json',dict(rows=rows,native=native,costs=costs));results={};lines=[
        'V25：命名/结构化指代表达/坐标生成阶段，与剪枝前视觉证据组合。',
        '56原训练RGB、36历史复用开发RGB（新24/旧12）；原视图、提议器、评分器结构固定。',
        '原6评分器冻结评估；另以相同训练样本/初始化/批次/64更新适配54小评分器，在开发新提示词特征提取前冻结。',
        '没有微调8B，没有使用新独立测试，没有HTML。']
    lookup={(r['sample_id'],r['regime'],r['model'],r['arm']):r for r in rows}
    for cohort in ['new_development','old_val','all']:
        cc=[c for c in cases if cohort=='all' or c['cohort']==cohort];summary={};comparisons={};lines+=['',f'{cohort}, n={len(cc)}']
        for regime in ['frozen','adapted']:
            summary[regime]={}
            for family in ['original3k','full']:
                aa=ALL_ARMS if regime=='frozen' else ARMS;summary[regime][family]={};arrays={}
                for arm in aa:
                    rr=[[lookup[c['sample_id'],regime,f'{family}_s{s}',arm] for c in cc] for s in [1,2,3]]
                    v=np.asarray([[r['hit'] for r in seed] for seed in rr],float);arrays[arm]=v
                    summary[regime][family][arm]=dict(accuracy_percent=100*float(v.mean()),seed_hits=v.sum(1).astype(int).tolist(),
                        oracle_percent=100*float(np.mean([r['oracle'] for seed in rr for r in seed])),
                        mean_candidates=float(np.mean([r['candidate_count'] for seed in rr for r in seed])),
                        raw_peak_any_percent=100*float(np.mean([r['raw_peak_any_hit'] for seed in rr for r in seed])),
                        by_source={src:100*float(v[:,[i for i,c in enumerate(cc) if c['source']==src]].mean()) for src in sorted({c['source'] for c in cc})})
                    m=summary[regime][family][arm];lines.append(f'{regime}/{family}/{arm}: {m["accuracy_percent"]:.2f}%,命中{m["seed_hits"]},覆盖{m["oracle_percent"]:.2f}%')
                for label,(a,b) in PAIRS.items():
                    level=1-.05/14 if cohort=='new_development' and regime=='adapted' else .95
                    comparisons[f'{regime}/{family}/{label}']=bootstrap(arrays[a]-arrays[b],cc,level)
        external={name:100*float(np.mean([hit(native[c['sample_id']][field],c,labels[c['sample_id']]) for c in cc]))
                  for name,field in [('coordinate_global','global_point'),('coordinate_consensus','consensus_point'),('MVP_cached','MVP')]}
        lines.append('显式坐标/固定视图一致性/MVP缓存：'+str(external))
        results[cohort]=dict(n=len(cc),methods=summary,comparisons=comparisons,external=external)
    gates={}
    for family in ['original3k','full']:
        for label in ['context_prompt','coordinate_task']:
            a=results['new_development']['comparisons'][f'adapted/{family}/{label}'];b=results['old_val']['comparisons'][f'adapted/{family}/{label}']
            gates[family+'/'+label]=a['delta_pp']>=3 and a['ci_pp'][0]>0 and b['delta_pp']>=-5
    lines+=['','适配后预登记信号门槛：'+str(gates),'新24为小型复用开发试验，不能作为新的独立泛化或MVP优势证明。',
        '结构化输出、原始attention加乘与几何位置先验分别比较。raw attention乘积不保证等于逻辑AND；未实现任意相对锚点关系。',
        'B/C允许96输出token、A历史64；实际耗时与token另列，不能宣称等计算。',
        f'原6评分器全部{parity}条A原流程预测一致。']
    write(OUT/'results.json',dict(complete=True,cohorts=results,positive_gates=gates,baseline_parity=parity,
        independent_test=False,mainline_unchanged=True,common_training_rgb=frozen['common_training_rgb']))
    (OUT/'results.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    write(OUT/'verification.json',dict(complete=True,baseline_parity=parity,prediction_rows=len(rows),
        source_sha256=source_identity(),protocol_sha256=sha(OUT/'protocol.json'),adapters=frozen))
    verify();paths=[OUT/n for n in ['protocol.json','results.json','results.txt','predictions.json','verification.json','models/frozen.json']]
    write(OUT/'completion.json',dict(complete=True,completed_at_utc=now(),artifact_sha256={str(p):sha(p) for p in paths},mainline_unchanged=True))
    reg=read(ROOT/'CURRENT_RESEARCH.json');before=read(OUT/'registry_before.json');assert reg['models']==before['models'] and reg['run_id']==before['run_id']
    reg['active_experiment'].update(status='complete',phase='inference_adaptation_and_analysis_complete',results_path=str(OUT/'results.txt'),completion_path=str(OUT/'completion.json'),positive_gates=gates)
    reg['completed_experiments'].append(dict(run_id=OUT.name,status='complete',analysis_path=str(OUT/'results.txt'),mainline_unchanged=True))
    write(ROOT/'CURRENT_RESEARCH.json',reg);status('complete',positive_gates=gates,prediction_rows=len(rows))
    print('\n'.join(lines),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['smoke','prepare','worker','evaluate','all'],default='all');args=parser.parse_args()
    try:
        if args.stage=='smoke':smoke();return
        if args.stage in ['prepare','all']:prepare()
        if args.stage in ['worker','all']:worker()
        if args.stage in ['evaluate','all']:evaluate()
    except Exception as exc:
        status('failed',error=str(exc),traceback=traceback.format_exc());raise


if __name__=='__main__':main()
