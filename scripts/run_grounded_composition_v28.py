"""Sequential V28 representation, grounded relations, and targeted supervision.

All checkpoints fixed before development scoring. No dashboard or OS action.
"""
import os
for k,v in dict(PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4').items():os.environ.setdefault(k,v)
from pathlib import Path
from datetime import datetime,timezone
from collections import Counter,defaultdict
import sys,json,hashlib,time,traceback,shutil,re,gc,argparse
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import numpy as np
import torch
from PIL import Image
from clickagent.grounded_composition_v28 import (GLOBAL_PROMPT,LOCAL_PROMPT,GroundedRuntime,visual_features,
    relation_features,fixed_crop,pixel,fit,infer,VIS_DIM)
from clickagent.evidence_retention import RetentionModel
from clickagent.model_features import ExtractionConfig
from run_evidence_retention_v14 import tensors
from run_visual_feedback_v23 import read,write,sha,hit,bootstrap
from run_prompt_readout_v25 import npz,load_npz,key,opened_view
from clickagent.visual_feedback_v23 import digest

OUT=ROOT/'runs/grounded_composition_v28';V19=ROOT/'runs/bottleneck_v19';V23=ROOT/'runs/visual_feedback_v23';V12=ROOT/'runs/crossfit_candidates_v12'
NAMES=[f'{f}_s{s}' for f in ['original3k','full'] for s in [1,2,3]]
SOURCES=[Path(__file__),ROOT/'src/clickagent/grounded_composition_v28.py',ROOT/'tests/test_grounded_composition_v28.py']
MODES=['raw','centered','uniform','shuffled']
STAGE2=['no_identity','no_relation','grounded','reversed_relation']
GROUPS=['random','targeted','combined','combined_nohard']

def now():return datetime.now(timezone.utc).isoformat()
def identity():return {str(p):sha(p) for p in SOURCES}
def status(phase,**kw):
    r=dict(phase=phase,updated_at_utc=now(),pid=os.getpid(),complete=phase=='complete',**kw)
    write(OUT/'status.json',r);print(json.dumps(r,ensure_ascii=False),flush=True)
def dest(c,part):return OUT/part/f'{key(c["sample_id"])}.npz'
def load(c,part):return load_npz(dest(c,part))
def verify():
    p=read(OUT/'protocol.json');assert p['source_sha256']==identity(),'V28 source drift'
    for path,h in p['sealed_sha256'].items():assert sha(path)==h,path
    return p


def prepare():
    assert not (OUT/'protocol.json').exists(),'Already registered'
    import subprocess
    check=subprocess.run([sys.executable,'-B','-X','utf8','-m','unittest','discover','-s','tests','-p','test_grounded_composition_v28.py','-v'],cwd=ROOT,capture_output=True,text=True,encoding='utf-8')
    write(OUT/'unit_tests.json',dict(exit_code=check.returncode,output=check.stdout+check.stderr,source_sha256=identity()))
    assert check.returncode==0,check.stdout+check.stderr
    indices=read(V19/'data/indices.json');old_eval=read(V19/'evaluation/train.json')
    lookup={}
    manifests=[ROOT/'runs/multiregion_v13/data/train.jsonl',ROOT/'runs/multiregion_v13/data/val.jsonl',ROOT/'runs/quality_readout_v15_3k/data/new_train.jsonl',ROOT/'runs/quality_readout_v15_3k/data/new_reserve.jsonl']
    for p in manifests:
        for line in p.read_text(encoding='utf-8-sig').splitlines():
            c=json.loads(line);lookup[c['sample_id']]=c
    train=indices['train'];by_source=defaultdict(list)
    for e in train:by_source[e['source']].append(e)
    failures={sid:sum(not old_eval['hit_vectors'][f'original_s{s}'][j] for s in [1,2,3]) for j,sid in enumerate(old_eval['sample_ids'])}
    selected=[];used=set();mining=[]
    relation_pattern=re.compile(r'\b(above|below|between|left of|right of|under|beneath|inside|within|next to)\b',re.I)
    for source,ee in sorted(by_source.items()):
        rr=[]
        for e in sorted(ee,key=lambda x:digest([x['group'],x['sample_id'],'v28_random'])):
            if e['group'] in used:continue
            rr.append(e);used.add(e['group'])
            if len(rr)==8:break
        assert len(rr)==8
        selected.extend((e,'random') for e in rr)
        eligible=[e for e in ee if e['group'] not in used and failures[e['sample_id']]>=2]
        # Interleave relation queries, reachable ranking failures, and missing coverage.
        bins=[[],[],[]]
        for e in eligible:
            relational=bool(relation_pattern.search(lookup[e['sample_id']]['instruction']))
            kind=0 if relational else (1 if e['matched_count'] else 2)
            bins[kind].append(e)
        bins=[sorted(b,key=lambda e:digest([e['sample_id'],'v28_targeted'])) for b in bins]
        hard=[]
        while len(hard)<8 and any(bins):
            for b in bins:
                while b and b[0]['group'] in used:b.pop(0)
                if b and len(hard)<8:
                    e=b.pop(0);used.add(e['group']);hard.append(e)
        assert len(hard)==8,(source,len(hard))
        selected.extend((e,'targeted') for e in hard)
    dev=read(V23/'data/public.json');dev_entries={e['sample_id']:e for co in ['old_val','new_development'] for e in indices[co]}
    cases=[];labels={};entries={}
    for e,group in selected:
        c=lookup[e['sample_id']];public={k:c[k] for k in ['sample_id','dataset','image','image_size','image_sha256','instruction']}
        public.update(cohort='train',source=e['source'],group=e['group'],source_group=e['source_group'],training_group=group)
        cases.append(public);labels[c['sample_id']]=c['target_geometry'];entries[c['sample_id']]=e
        mining.append(dict(sample_id=c['sample_id'],group=group,source=e['source'],original_failed_seeds=failures[c['sample_id']],original_pool_covered=bool(e['matched_count']),explicit_relation=bool(relation_pattern.search(c['instruction']))))
    for public in dev:
        c=lookup[public['sample_id']];cases.append(dict(public,training_group=None));labels[c['sample_id']]=c['target_geometry'];entries[c['sample_id']]=dev_entries[c['sample_id']]
    for c in cases:
        e=entries[c['sample_id']];p=read(e['pipeline_path'])
        c['views']=[dict(feature_path=p['global_feature_path'],feature_sha256=p['global_feature_sha256'],transform=None)]
        c['views'] += [{k:v[k] for k in ['feature_path','feature_sha256','transform']} for v in p['local_views']]
        assert sha(c['image'])==c['image_sha256']
    assert len(cases)==212 and len({c['group'] for c in cases})==212
    assert not {c['group'] for c in cases if c['cohort']=='train'}&{c['group'] for c in dev}
    cases.sort(key=lambda c:(c['cohort']!='train',digest([c['sample_id'],'v28_order'])))
    for name,data in [('public',cases),('labels',labels),('indices',entries),('training_mining',mining)]:write(OUT/f'data/{name}.json',data)
    dependencies=[ROOT/'src/clickagent'/f'{n}.py' for n in ['evidence_retention','conditioned_features','target_token','model_features','action_model']]
    sealed=SOURCES+dependencies+manifests+[V19/'data/indices.json',V19/'evaluation/train.json',V23/'protocol.json',V23/'data/public.json',V12/'runtime_identity.json']+[OUT/f'data/{n}.json' for n in ['public','labels','indices','training_mining']]
    spec=dict(registered_at_utc=now(),source_sha256=identity(),sealed_sha256={str(p):sha(p) for p in sealed},models=read(V23/'protocol.json')['models'],
        data='112 original-training RGB: random56(8/source7), targeted56(8/source7), disjoint RGB. Targeted selected using ONLY historic original3k TRAIN failures >=2/3 fits, interleaving explicit relations, reachable mistakes, missing coverage. 100 reused development RGB(new72,old28), never used for training/mining/normalization. This is mechanistic development, NOT independent confirmation.',
        stage1='Fixed32-point A pool: protect top1 of ALL6 old heads, fill by mean standardized old scores. Compute original raw vs image-mean-subtracted BEFORE unit/attention pooling. Keep first4 actual token slots plus all-token mean/min, all96 heads, global separate from local; locals average only within local side. Uniform/spatially shuffled attention same capacity. Train random56, fixed ridge.3, all6 checkpoints, frozen before development. No token/head grouping.',
        stage2='Fresh frozen8B own-forward global structured target+point and up to2 explicit named reference OBJECT boxes/relations. Relation types left/right/above/below/inside/near; between becomes2 predicates. No reference invented if absent from instruction. Candidate-driven 640x640 max original-pixel crop centered on predicted global point, fallback frozen A top1 consensus, fallback image center, resized2x. Two NEW local calls on EXACT SAME crop: free target vs global-target-locked instruction. Local calls never execute OS actions. Full features use global+locked appearance, no_identity uses global+free. Relation feature evaluates nonlinear sigmoids against predicted object BOX EDGES, conjunction product. Reversed-direction control. Not a centroid proxy. Anchor box accuracy cannot be verified without anchor annotations.',
        calls='212 global160-token cap +424 local56-token cap =636 planned VLM calls. Same frozen Qwen3-VL-8B-Instruct runtime contract/layers/pixel cap/greedy. Actual counts/time reported. All arms share collected evidence; cost is NOT equal to original A or MVP. Role spans use actual generated prefix positions and own-token forwards.',
        stage2_pool='Same fixed A32 plus up to3 valid native target points(global/free/locked), stable dedup. Maximum35 common candidates, no GT points. New-coordinate prior is median A logit for each old head; if A empty all base logits0. Explicit new-point provenance included. This is a neutral prior, not an original-head prediction at unseen points.',
        stage3='Exactly same grounded feature/model/loss/ridge: random56 vs targeted56 (equal independent query count), combined112; combined_nohard keeps high-score negatives only. Hard pairing: top4 positive candidates x union nearest8 false candidates to positive region and highest-score8 negatives. No GT is ever supplied to inference, crop planning or candidate construction. No annotated point/box added to inference pool. Labels are used for TRAIN sampling/pairing only. Empty/no-positive pools remain counted and get no pairwise gradient.',
        learning='Frozen old6 heads, proposer, backbone. Linear residual with per-channel TRAIN std floor1e-4, fixed ridge.3 (no grid/search/development selection), query/source-balanced pairwise logistic. All fits report 3 deterministic source-stratified RGB folds as train-only diagnostic, never checkpoint selection. Each old checkpoint paired with all arms. Old3k/full refer only old-head training size; new fit sizes are explicitly56/112.',
        contrasts=[['repr_centered','repr_raw'],['repr_centered','repr_uniform'],['repr_centered','repr_shuffled'],['grounded','no_identity'],['grounded','no_relation'],['grounded','reversed_relation'],['targeted','grounded'],['combined','grounded'],['combined','combined_nohard'],['combined','A']],
        statistics='new72 primary descriptive effect estimates with5000 source-stratified RGB bootstrap,3 fixed-fit means. 20 comparisons across2 old-head families =>99.75% intervals;95% for old28/source summaries. Neither seeds nor views treated as independent tasks. No automatic adoption; all arms/results reported. Positive development evidence only if corrected lower>0; no MVP superiority claim or independent test in this pilot.',
        user_authorization='按照你说的做吧，三个环节都进行',outputs='Python logs, JSON/TXT/NPZ/PT/PNG; no HTML',mainline_unchanged=True)
    for row in spec['models'].values():assert sha(row['path'])==row['sha256']
    write(OUT/'registry_before.json',read(ROOT/'CURRENT_RESEARCH.json'));write(OUT/'protocol.json',spec)
    for p in SOURCES:
        target=OUT/'source_archive'/p.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,target)
    reg=read(ROOT/'CURRENT_RESEARCH.json');reg['active_experiment']=dict(run_id=OUT.name,status='running',phase='registered',protocol_path=str(OUT/'protocol.json'),status_path=str(OUT/'status.json'),mainline_unchanged=True)
    write(ROOT/'CURRENT_RESEARCH.json',reg)
    write(ROOT/'research/decisions/20261109_grounded_composition_v28.json',dict(registered_at_utc=now(),user_authorization=spec['user_authorization'],protocol_sha256=sha(OUT/'protocol.json'),mainline_unchanged=True))
    status('registered',train_rgb=112,dev_rgb=100,vlm_calls=636)


def score_original(cases):
    torch.set_num_threads(4);spec=verify();models={}
    for name,r in spec['models'].items():
        saved=torch.load(r['path'],map_location='cpu',weights_only=True);m=RetentionModel().eval().requires_grad_(False);m.load_state_dict(saved['state_dict']);models[name]=(m,saved)
    entries=read(OUT/'data/indices.json');hashes={}
    for ci,c in enumerate(cases,1):
        if dest(c,'original').exists():continue
        e=entries[c['sample_id']]
        for f,h in [('path','sha256'),('state_path','state_sha256'),('mask_path','mask_sha256'),('pipeline_path','pipeline_sha256')]:
            assert sha(e[f])==e[h];hashes[e[f]]=e[h]
        with np.load(e['path'],allow_pickle=False) as z:r={k:z[k].copy() for k in ['points','base','active','baseline']}
        r.update(sample_id=c['sample_id'],n=len(r['points']),state=np.load(e['state_path']),mask=np.load(e['mask_path']))
        scores=[]
        for name in NAMES:
            model,saved=models[name]
            if r['n']:
                inputs,_=tensors([r],saved['mean'].numpy(),saved['std'].numpy(),'aligned','cpu')
                with torch.inference_mode():s=model(**inputs).numpy()
            else:s=np.empty(0)
            scores.append(s)
        scores=np.asarray(scores);n=r['n'];ix=[]
        if n:
            for s in scores:
                j=int(s.argmax())
                if j not in ix:ix.append(j)
            z=(scores-scores.mean(1,keepdims=True))/np.maximum(scores.std(1,keepdims=True),1e-6)
            for j in np.argsort(-z.mean(0),kind='stable'):
                if len(ix)>=min(32,n):break
                if int(j) not in ix:ix.append(int(j))
        ix=np.asarray(sorted(ix),int)
        npz(dest(c,'original'),points=r['points'][ix],scores=scores[:,ix],complete_points=r['points'],complete_scores=scores,
            metadata=np.asarray(json.dumps(dict(sample_id=c['sample_id'],indices=ix.tolist(),labels_used=False))))
        if ci%8==0 or ci==len(cases):status('original_scoring',completed=ci,total=len(cases))
    write(OUT/'original_inputs_sha256.json',hashes)


def aggregate(c,points,mode,observations=None):
    result=np.zeros((len(points),VIS_DIM),np.float32);counts=np.zeros((len(points),2),int)
    if observations is None:
        observations=[]
        for vi,v in enumerate(c['views']):
            assert sha(v['feature_path'])==v['feature_sha256']
            with np.load(v['feature_path'],allow_pickle=False) as z:
                m=json.loads(str(z['metadata']));observations.append((z['target_attention'],z['vision_hidden'],m,v['transform'],vi>0))
    for att,vision,meta,transform,local in observations:
        f,valid=visual_features(att,vision,meta['grid_shape'],points,c['image_size'],transform,mode,c['sample_id']+str(transform))
        col=int(local);result[:,col*576:(col+1)*576]+=f;counts[:,col]+=valid
    for col in [0,1]:result[:,col*576:(col+1)*576]/=np.maximum(counts[:,col,None],1)
    return result,counts>0


def build_representation(cases):
    for ci,c in enumerate(cases,1):
        if dest(c,'representation').exists():continue
        original,_=load(c,'original');points=original['points'];values={}
        for mode in MODES:values[mode],valid=aggregate(c,points,mode)
        npz(dest(c,'representation'),points=points,valid=valid,**values,metadata=np.asarray(json.dumps(dict(sample_id=c['sample_id'],labels_used=False))))
        if ci%4==0 or ci==len(cases):status('representation_'+c['cohort'],completed=ci,total=len(cases))


def record(c,model,arm,labels=None):
    original,_=load(c,'original');si=NAMES.index(model)
    if arm.startswith('repr_'):
        a,_=load(c,'representation');points=a['points'];s=original['scores'][si];x=a[arm[5:]]
    else:
        a,_=load(c,'grounded');points=a['points'];s=a['scores'][si];which='free' if arm=='no_identity' else 'locked'
        x=np.concatenate([a[which],a['geometry_'+which],a['reversed_relation'] if arm=='reversed_relation' else (np.zeros_like(a['relation']) if arm=='no_relation' else a['relation'])],1)
    base=(s-s.mean())/max(float(s.std()),1e-6) if len(s) else s
    r=dict(sample_id=c['sample_id'],source=c['source'],x=x.astype(np.float64),base=base.astype(np.float64),points=points,size=np.asarray(c['image_size']),training_group=c.get('training_group'))
    if labels is not None:r['y']=np.asarray([hit(p.tolist(),c,labels[c['sample_id']]) for p in points],bool)
    return r


def training(cases,stage):
    labels=read(OUT/'data/labels.json');assert all(c['cohort']=='train' for c in cases)
    arms=['repr_'+m for m in MODES] if stage==1 else [*STAGE2,'targeted','combined','combined_nohard']
    frozen={};folds={}
    for src in sorted({c['source'] for c in cases}):
        for j,c in enumerate(sorted([c for c in cases if c['source']==src],key=lambda c:digest([c['group'],'v28fold']))):folds[c['sample_id']]=j%3
    for arm in arms:
        subset=[c for c in cases if arm in ['combined','combined_nohard'] or c['training_group']==('targeted' if arm=='targeted' else 'random')]
        for model in NAMES:
            rr=[record(c,model,arm,labels) for c in subset];hard=arm!='combined_nohard';oof=[]
            for fold in range(3):
                fit_rr=[r for r in rr if folds[r['sample_id']]!=fold];val=[r for r in rr if folds[r['sample_id']]==fold]
                fitted=fit(fit_rr,.3,hard)
                oof.extend(dict(sample_id=r['sample_id'],source=r['source'],hit=bool(r['y'][int(infer(fitted,r).argmax())]) if len(r['x']) else False) for r in val)
            fitted=fit(rr,.3,hard);target=OUT/f'models_stage{stage}/{arm}__{model}.npz'
            npz(target,weight=fitted['weight'],mean=fitted['mean'],std=fitted['std'],metadata=np.asarray(json.dumps(dict(arm=arm,old_head=model,new_training_rgb=len(subset)))))
            info=dict(path=str(target),sha256=sha(target),new_training_rgb=len(subset),training_ids=[c['sample_id'] for c in subset],oof=oof,
                train_hits=sum(bool(r['y'][int(infer(fitted,r).argmax())]) if len(r['x']) else False for r in rr),oof_hits=sum(r['hit'] for r in oof),
                optimizer={k:v for k,v in fitted.items() if k not in ['weight','mean','std']},development_used=False)
            frozen[arm+'__'+model]=info
            status('training_stage'+str(stage),arm=arm,model=model,train_rgb=len(subset),train_hits=info['train_hits'],oof_hits=info['oof_hits'],completed=len(frozen),total=len(arms)*6)
    write(OUT/f'models_stage{stage}/frozen.json',dict(complete=True,at_utc=now(),models=frozen,development_used=False))


def load_runtime():
    from clickagent.advantage_runtimes import configure_paths
    configure_paths('coordinate');torch.set_num_threads(4)
    cfg=dict(read(V12/'runtime_identity.json')['runtime_contract']);cfg['layers']=tuple(cfg['layers']);cfg['conditions']=tuple(cfg['conditions'])
    cfg['max_memory']={int(k) if str(k).isdigit() else k:v for k,v in cfg['max_memory'].items()};cfg['offload_folder']=str(OUT/'offload')
    return GroundedRuntime(ExtractionConfig(**cfg))


def collect(cases):
    missing=[c for c in cases if not all(dest(c,'raw/'+part).exists() for part in ['global','free','locked'])]
    if not missing:return
    status('loading_frozen_8b',remaining_rgb=len(missing));runtime=load_runtime();write(OUT/'runtime_metadata.json',runtime.runtime_metadata)
    completed=3*(len(cases)-len(missing));total=3*len(cases)
    for c in missing:
        sid=c['sample_id']
        for part in ['global','free','locked']:
            target=dest(c,'raw/'+part)
            if target.exists():completed+=1;continue
            if part=='global':
                transform=None;prompt=GLOBAL_PROMPT;instruction=c['instruction'];cap=160
            else:
                _,gm=load(c,'raw/global');transform=gm['planned_local_transform'];prompt=LOCAL_PROMPT;instruction=c['instruction'];cap=56
                if part=='locked':instruction+='\nResolved target identity from the full screenshot: '+(gm['target'] or 'UNKNOWN')+'. Locate THIS SAME referent in this crop. Keep the original instruction constraints; use NONE if it is absent.'
            im=opened_view(c,dict(transform=transform));a,m=runtime.infer(im,instruction,prompt,cap);im.close()
            m.update(sample_id=sid,part=part,transform=transform,image_sha256=c['image_sha256'],instruction=instruction,protocol_sha256=sha(OUT/'protocol.json'))
            if part=='global':
                point=pixel(m['point'],c['image_size'])
                if point is None:
                    original,_=load(c,'original');s=original['scores'];pp=original['points']
                    point=pp[int(((s-s.mean(1,keepdims=True))/np.maximum(s.std(1,keepdims=True),1e-6)).mean(0).argmax())].tolist() if len(pp) else [c['image_size'][0]//2,c['image_size'][1]//2]
                m['planned_local_transform']=fixed_crop(point,c['image_size'])
            npz(target,metadata=np.asarray(json.dumps(m,ensure_ascii=False)),**a);completed+=1
            status('collecting_'+c['cohort'],sample_id=sid,part=part,completed_calls=completed,total_calls=total,elapsed_seconds=m['elapsed_seconds'],tokens=len(m['generated_token_ids']),parse_complete=m['complete'])
    del runtime;gc.collect();torch.cuda.empty_cache()
    status('fresh_observations_complete',calls=total)


def role_observation(a,m,transform,local):
    indices=np.asarray(m['roles'].get('target',[]),int)-1
    return a['all_attention'][indices],a['vision_hidden'],m,transform,local


def build_grounded(cases):
    reverse=dict(left='right',right='left',above='below',below='above',inside='near',near='inside')
    for ci,c in enumerate(cases,1):
        if dest(c,'grounded').exists():continue
        original,_=load(c,'original');raw={part:load(c,'raw/'+part) for part in ['global','free','locked']}
        gp=raw['global'][1];transform=gp['planned_local_transform'];native={}
        for part,(a,m) in raw.items():native[part]=pixel(m['point'],m['image_size'],m['transform'])
        pp=original['points'].tolist();origins=[[1,0,0,0] for _ in pp]
        for j,part in enumerate(['global','free','locked'],1):
            p=native[part]
            if p is not None:
                if p not in pp:pp.append(p);origins.append([0,0,0,0])
                origins[pp.index(p)][j]=1
        points=np.asarray(pp,int).reshape(-1,2);n=len(points);oldn=len(original['points']);scores=np.zeros((6,n))
        for si,s in enumerate(original['scores']):scores[si]=np.r_[s,np.full(n-oldn,float(np.median(s)) if len(s) else 0.)]
        values={};global_ob=role_observation(*raw['global'],None,False)
        for which in ['free','locked']:
            local_ob=role_observation(*raw[which],transform,True)
            values[which],valid=aggregate(c,points,'centered',[global_ob,local_ob])
            geometry=np.zeros((n,17),np.float32);geometry[:,:4]=np.asarray(origins).reshape(-1,4)
            if which=='free':geometry[:,3]=geometry[:,2]  # do not reveal locked coordinate to free arm
            xy=points/np.asarray(c['image_size'])
            for j,part in enumerate(['global',which]):
                p=native[part]
                if p is not None:
                    d=xy-np.asarray(p)/np.asarray(c['image_size']);geometry[:,4+j*5:9+j*5]=np.c_[d,np.linalg.norm(d,axis=1),np.exp(-np.sum(d*d,axis=1)/(2*.03**2)),np.ones(n)]
            geometry[:,14:16]=valid;geometry[:,16]=int(gp['target'] is not None)
            values['geometry_'+which]=geometry
        anchors=gp['anchors'];values['relation']=relation_features(points,c['image_size'],anchors)
        rev=[dict(a,relation=reverse.get(a['relation'],a['relation'])) for a in anchors];values['reversed_relation']=relation_features(points,c['image_size'],rev)
        npz(dest(c,'grounded'),points=points,scores=scores,**values,metadata=np.asarray(json.dumps(dict(sample_id=c['sample_id'],native_points=native,global_target=gp['target'],free_target=raw['free'][1]['target'],locked_target=raw['locked'][1]['target'],anchors=anchors,labels_used=False))))
        if ci%4==0 or ci==len(cases):status('building_grounded_'+c['cohort'],completed=ci,total=len(cases))


def evaluate(cases,stage):
    frozen=read(OUT/f'models_stage{stage}/frozen.json');assert frozen['complete'];labels=read(OUT/'data/labels.json');rows=[];coverage=[];diagnostics=[]
    models={}
    for name,r in frozen['models'].items():assert sha(r['path'])==r['sha256'];models[name]=load_npz(r['path'])[0]
    for ci,c in enumerate(cases,1):
        original,_=load(c,'original');truth=np.asarray([hit(p.tolist(),c,labels[c['sample_id']]) for p in original['points']],bool)
        for si,name in enumerate(NAMES):
            s=original['scores'][si];j=int(s.argmax()) if len(s) else None
            rows.append(dict(sample_id=c['sample_id'],old_head=name,arm='A',point=original['points'][j].tolist() if j is not None else None,hit=bool(j is not None and truth[j])))
        coverage.append(dict(sample_id=c['sample_id'],A32=bool(truth.any()),A_complete=any(hit(p.tolist(),c,labels[c['sample_id']]) for p in original['complete_points'])))
        for tag,m in models.items():
            arm,name=tag.split('__');r=record(c,name,arm);j=int(infer(m,r).argmax()) if len(r['x']) else None;p=r['points'][j].tolist() if j is not None else None
            rows.append(dict(sample_id=c['sample_id'],old_head=name,arm=arm,point=p,hit=hit(p,c,labels[c['sample_id']])))
        if stage==2:
            a,m=load(c,'grounded');coverage[-1]['augmented']=any(hit(p.tolist(),c,labels[c['sample_id']]) for p in a['points'])
            for arm,p in m['native_points'].items():rows.append(dict(sample_id=c['sample_id'],old_head='native',arm='native_'+arm,point=p,hit=hit(p,c,labels[c['sample_id']])))
            diagnostics.append(dict(sample_id=c['sample_id'],**m))
        if ci%10==0 or ci==len(cases):status('evaluation_stage'+str(stage),completed=ci,total=len(cases))
    write(OUT/f'predictions_stage{stage}.json',dict(rows=rows,coverage=coverage,diagnostics=diagnostics))


def analyze():
    cases=read(OUT/'data/public.json');dev=[c for c in cases if c['cohort']!='train'];spec=verify();p1=read(OUT/'predictions_stage1.json');p2=read(OUT/'predictions_stage2.json')
    lookup={(r['sample_id'],r['old_head'],r['arm']):r for r in p1['rows']+p2['rows']};arms=['A',*['repr_'+m for m in MODES],*STAGE2,'targeted','combined','combined_nohard'];results={};lines=['V28 三环节实验结果','性质：复用开发集机制实验，不是独立确认；旧评分器/主干保持冻结。','新增模块训练 random56 / targeted56 / combined112；3k/full仅指旧评分器。','']
    for cohort in ['new_development','old_val','all']:
        cc=[c for c in dev if cohort=='all' or c['cohort']==cohort];summary={};contrasts={};lines.append(f'{cohort}: {len(cc)} RGB')
        for family in ['original3k','full']:
            summary[family]={};arrays={}
            for arm in arms:
                v=np.asarray([[lookup[c['sample_id'],f'{family}_s{s}',arm]['hit'] for c in cc] for s in [1,2,3]],float);arrays[arm]=v
                summary[family][arm]=dict(percent=float(v.mean()*100),seed_hits=v.sum(1).astype(int).tolist(),by_source={src:100*float(v[:,[j for j,c in enumerate(cc) if c['source']==src]].mean()) for src in sorted({c['source'] for c in cc})})
            for a,b in spec['contrasts']:contrasts[family+'/'+a+'-'+b]=bootstrap(arrays[a]-arrays[b],cc,.9975 if cohort=='new_development' else .95)
            lines.append(f'  {family}: '+', '.join(f'{a}={summary[family][a]["percent"]:.2f}' for a in arms))
        native={part:100*float(np.mean([lookup[c['sample_id'],'native','native_'+part]['hit'] for c in cc])) for part in ['global','free','locked']}
        coverage={k:100*float(np.mean([r[k] for r in p2['coverage'] if r['sample_id'] in {c['sample_id'] for c in cc}])) for k in ['A_complete','A32','augmented']}
        results[cohort]=dict(summary=summary,contrasts=contrasts,native=native,coverage=coverage);lines.extend(['  native: '+str(native),'  coverage: '+str(coverage),''])
    new_ids={c['sample_id'] for c in dev if c['cohort']=='new_development'};di=[r for r in p2['diagnostics'] if r['sample_id'] in new_ids]
    counts=dict(grounded_reference_queries=sum(any(a['valid'] for a in r['anchors']) for r in di),two_reference_queries=sum(sum(a['valid'] for a in r['anchors'])==2 for r in di),global_none=sum(r['global_target'] is None for r in di),free_name_changed=sum(r['free_target']!=r['global_target'] for r in di),locked_name_changed=sum(r['locked_target']!=r['global_target'] for r in di))
    results['mechanisms']=counts;lines+=['新72机制统计（名称差异不自动判错）: '+str(counts)]
    training={}
    for stage in [1,2]:
        frozen=read(OUT/f'models_stage{stage}/frozen.json');training.update({k:{f:r[f] for f in ['new_training_rgb','train_hits','oof_hits','optimizer']} for k,r in frozen['models'].items()})
    results['training']=training
    cost=[]
    for c in cases:
        for part in ['global','free','locked']:
            with np.load(dest(c,'raw/'+part),allow_pickle=False) as z:m=json.loads(str(z['metadata']))
            cost.append(dict(sample_id=c['sample_id'],part=part,seconds=m['elapsed_seconds'],tokens=len(m['generated_token_ids']),complete=m['complete'],capped=m['token_cap_reached']))
    results['cost']=dict(actual_calls=len(cost),seconds=sum(r['seconds'] for r in cost),tokens=sum(r['tokens'] for r in cost),parse_failures=sum(not r['complete'] for r in cost),capped=sum(r['capped'] for r in cost))
    lines+=['调用成本: '+str(results['cost']),'','未将任何开发分数用于模型/种子/训练样本/裁剪/候选选择。所有拟合固定ridge=.3，没有按开发集挑最优分支。','主线保持V15_3k；本轮没有MVP直接同成本/独立测试，不能据此声称超过MVP。']
    write(OUT/'results.json',results);(OUT/'results.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8');status('analysis_complete',results_path=str(OUT/'results.txt'))


def audit():
    spec=verify();cases=read(OUT/'data/public.json');checks=0;hashes={}
    for c in cases:
        for part in ['global','free','locked']:
            f=dest(c,'raw/'+part);a,m=load_npz(f);hashes[str(f)]=sha(f)
            assert m['protocol_sha256']==sha(OUT/'protocol.json') and m['sample_id']==c['sample_id'];assert m['ground_truth_in_model_input'] is False
            assert m['own_token_forward'] and m['all_backbone_parameters_frozen']
            assert len(a['all_attention'])==len(a['all_hidden'])==len(m['generated_token_ids'])
            from clickagent.grounded_composition_v28 import parse
            parsed=parse(m['generated_text'],m['generated_prefixes']);assert all(parsed[k]==m[k] for k in ['roles','point','anchors','target','complete'])
            assert all(1<=i<=len(m['generated_token_ids']) for rr in m['roles'].values() for i in rr);checks+=1
        _,free=load(c,'raw/free');_,locked=load(c,'raw/locked');assert free['transform']==locked['transform'];assert free['system_prompt']==locked['system_prompt']
    for stage in [1,2]:
        frozen=read(OUT/f'models_stage{stage}/frozen.json');assert not frozen['development_used']
        for tag,row in frozen['models'].items():assert sha(row['path'])==row['sha256'];assert set(row['training_ids'])<={c['sample_id'] for c in cases if c['cohort']=='train'}
    for r in spec['models'].values():assert sha(r['path'])==r['sha256']
    write(OUT/'audit.json',dict(passed=True,at_utc=now(),own_token_trajectories=checks,same_local_crop=True,models_frozen_before_development=True,raw_sha256=hashes,mainline_weights_unchanged=True))
    artifacts=[p for p in OUT.rglob('*') if p.is_file() and p.suffix in ['.json','.txt','.npz'] and 'offload' not in p.parts and p.name not in ['status.json','completion.json']]
    write(OUT/'completion.json',dict(complete=True,at_utc=now(),artifact_sha256={str(p):sha(p) for p in artifacts},mainline_unchanged=True))
    reg=read(ROOT/'CURRENT_RESEARCH.json');reg['active_experiment'].update(status='complete',phase='all_three_stages_analyzed_and_audited',results_path=str(OUT/'results.txt'),completion_path=str(OUT/'completion.json'),audit_path=str(OUT/'audit.json'))
    write(ROOT/'CURRENT_RESEARCH.json',reg);status('complete',stages_completed=[1,2,3],trajectories=checks,results_path=str(OUT/'results.txt'))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['prepare','stage1','collect_train','fit','collect_dev','finish','all']);arg=parser.parse_args().phase
    if arg in ['prepare','all']:prepare()
    verify();cases=read(OUT/'data/public.json');train=[c for c in cases if c['cohort']=='train'];dev=[c for c in cases if c['cohort']!='train']
    if arg in ['stage1','all']:
        score_original(cases);build_representation(train);training(train,1);build_representation(dev);evaluate(dev,1)
    if arg in ['collect_train','all']:collect(train);build_grounded(train)
    if arg in ['fit','all']:training(train,2)
    if arg in ['collect_dev','all']:
        assert read(OUT/'models_stage2/frozen.json')['complete'];collect(dev);build_grounded(dev)
    if arg in ['finish','all']:evaluate(dev,2);analyze();audit()

if __name__=='__main__':
    try:main()
    except BaseException as e:
        status('failed',error=repr(e),traceback=traceback.format_exc());raise
