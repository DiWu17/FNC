"""Sequential V31: shorten global output, diversify ROI, learn real-content utility."""
import os
for k,v in dict(PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
               OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4').items():os.environ.setdefault(k,v)
import argparse
from collections import Counter,defaultdict
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import numpy as np
from PIL import Image
from clickagent.reinspection_v31 import (SHORT_GLOBAL_PROMPT,JUDGE_PROMPT,MODES,key,unit,plan_diverse,
    proposal_observation,first_valid,consensus,eligible_pair,judge_canvas,pair_geometry,joint_vector,
    fit,scores,decide)
OUT=ROOT/'runs/reinspection_v31';PRIOR=ROOT/'runs/grounded_composition_v28'
SOURCES=[Path(__file__),ROOT/'src/clickagent/reinspection_v31.py',ROOT/'tests/test_reinspection_v31.py']

def now():return datetime.now(timezone.utc).isoformat()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def default(v):
    if isinstance(v,np.ndarray):return v.tolist()
    if isinstance(v,np.generic):return v.item()
    raise TypeError(type(v).__name__)
def replace(a,b):
    for i in range(100):
        try:os.replace(a,b);return
        except PermissionError:
            if i==99:raise
            time.sleep(.1)
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(v,ensure_ascii=False,indent=2,default=default,allow_nan=False)+'\n',encoding='utf-8');replace(tmp,p)
def npz(p,arrays,metadata):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp')
    with tmp.open('wb') as f:np.savez_compressed(f,metadata=np.asarray(json.dumps(metadata,ensure_ascii=False,default=default)),**arrays)
    replace(tmp,p)
def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files if k!='metadata'},json.loads(str(z['metadata']))
def path(c,part):return OUT/part/(key(c['sample_id'])+'.npz')
def status(phase,**kw):
    v=dict(phase=phase,updated_at_utc=now(),pid=os.getpid(),complete=phase=='complete',**kw)
    write(OUT/'status.json',v);print(json.dumps(v,ensure_ascii=False,default=default),flush=True)
def verify():
    p=read(OUT/'protocol.json')
    for name,h in p['hashes'].items():assert sha(name)==h,name
    for e in p['backbone_inventory']:
        s=Path(e['path']).stat();assert s.st_size==e['bytes'] and s.st_mtime_ns==e['mtime_ns']
    return p

def prepare():
    if (OUT/'protocol.json').exists():return verify()
    OUT.mkdir(parents=True,exist_ok=True)
    test=subprocess.run([sys.executable,'-B','-X','utf8','-m','unittest','discover','-s','tests','-p','test_reinspection_v31.py','-v'],cwd=ROOT,capture_output=True,text=True,encoding='utf-8')
    write(OUT/'unit_tests.json',dict(exit_code=test.returncode,output=test.stdout+test.stderr));assert test.returncode==0,test.stdout+test.stderr
    original=read(PRIOR/'data/public.json');cases=[{k:c[k] for k in ['sample_id','dataset','image','image_size','image_sha256','instruction','cohort','source','group','source_group','training_group']} for c in original]
    assert Counter(c['cohort'] for c in cases)==dict(train=112,old_val=28,new_development=72)
    assert len({c['group'] for c in cases})==212
    labels=read(PRIOR/'data/labels.json');assert set(labels)=={c['sample_id'] for c in cases}
    write(OUT/'data/public.json',cases);write(OUT/'data/labels.json',labels)
    timing=sorted([c for c in cases if c['cohort']=='train'],key=lambda c:key(c['sample_id']+'|v31_prompt_timing'))[:6]
    write(OUT/'data/prompt_timing.json',[c['sample_id'] for c in timing])
    frozen_inputs=SOURCES+[ROOT/'src/clickagent/grounded_composition_v28.py',ROOT/'src/clickagent/crossfit_proposer.py',
        ROOT/'src/clickagent/conditioned_features.py',ROOT/'src/clickagent/action_model.py',ROOT/'src/clickagent/model_features.py',
        ROOT/'src/clickagent/public_model.py',ROOT/'src/clickagent/public_evaluation.py',
        *[ROOT/'src/clickagent'/f'{name}.py' for name in ['target_token','attention_features','candidate_ranker','semantic_head_ranker','attention_field_ranker','point_aligned_readout']],
        ROOT/'runs/crossfit_candidates_v12/runtime_identity.json',ROOT/'runs/crossfit_candidates_v12/proposers/full.pt',
        PRIOR/'data/public.json',PRIOR/'data/labels.json',PRIOR/'protocol.json',
        OUT/'data/public.json',OUT/'data/labels.json',OUT/'data/prompt_timing.json',OUT/'unit_tests.json']
    baseline_index={};mvp_index={}
    mvp_cases={c['sample_id']:c for c in read(ROOT/'runs/mvp_repair_v21/data/public.json')}
    for c in cases:
        for part in ['global','free']:
            p=PRIOR/'raw'/part/(key(c['sample_id'])+'.npz');assert p.exists();baseline_index[str(p)]=sha(p)
        if c['cohort']!='train':
            old=mvp_cases[c['sample_id']]
            assert all(c[k]==old[k] for k in ['instruction','image_sha256','image_size'])
            p=ROOT/'runs/mvp_repair_v21/mvp'/(key(c['sample_id'])+'.json');assert p.exists();mvp_index[str(p)]=sha(p)
    write(OUT/'baseline_input_index.json',baseline_index);write(OUT/'mvp_input_index.json',mvp_index)
    frozen_inputs += [OUT/'baseline_input_index.json',OUT/'mvp_input_index.json',ROOT/'runs/mvp_repair_v21/data/public.json']
    spec=dict(registered_at_utc=now(),user_authorization='试试看吧',
        scope='Mechanism pilot on SAME112training/100reused-development RGB from V28;not independent benchmark confirmation. Preserve adopted V15_3k and all checkpoints.',
        observations='Frozen Qwen3-VL-8B FP16,layers15/28/36,max_pixels262144. New short global XML(t,p only),cap64. Original LOCAL_PROMPT cap56. No teacher coordinates/names or target boxes in inference.',
        roi='Global name-token observations -> frozen611919-param proposer,top64 distinct candidate coordinates -> select640x640 max crop maximizing spatial score mass withIoU<=.25 vs first crop if feasible. First crop centered on short native global, fallback proposal top1, fallback image center. Equal-cost second-view control:640 crop recentered on first local prediction. Both resize2x and use full original instruction. Four fresh generated views perRGB:global,local0,refine,diverse. Even identical crop inputs are rerun,not cached.',
        candidates='Keep exact native coordinates. Base=first valid(local0,global,proposal top1). Alternative=first valid(diverse-local,global). Equal-cost simple consensus compares global/local0/refine vs global/local0/diverse. Native union coverage is diagnostic oracle,never used for inference. Historical A-pool scores not used for new ROI or selection.',
        content='For each distinct valid base/alternative pair,compose512x512 image:top full screenshot overview512x256,bottom2 original128-pixel crops enlarged2x;points shown as hollow rings. Full original instruction is text input. Frozen VLM single prefill,no answer generation/teacher forcing;read4096D last normalized decoder token before LM head. Both candidate orders;feature=unit(unit(h_forward)-unit(h_reverse)),preserving original4096 channels,no64D/random projection. Matched image-content control blurs GUI pixels before drawing identical marks/headings;same source/crops/masks/budget/prompt/architecture. Full/blurred inputs each in both orders.',
        training='Fixed weighted kernel ridge(lambda.01),all eligible112trainingqueries including utility0 (both correct/both wrong),target=hit(alternative)-hit(base). Source/query balanced. Content-only,geometry-only,half-weight content+geometry,matchedblurred+geometry,and geometry+cyclicallyshuffled joint embeddings. No hyperparameter search. Strict predicted utility>.1 main switching rule;>.0 exploratory. Utility is not a correctness probability. If base absent,valid alternative fallback;no valid alternative keeps base.',
        cv='Source_group-disjoint3fold and7source-LOSO,only新增utilitymodule heldout;proposer trained on original training sources. All observations/labels for fitting ONLY training112. Freeze all finalheads before ANYnewdevelopmentobservation or scoring. Reused development is diagnostic,not independent generalization evidence.',
        primary_contrasts=[['diverse_union','refine_union'],['combined','short_single'],['combined','geometry'],['combined','blurred'],['combined','shuffled']],
        statistics='All100 primary;5000source-stratified RGB cluster bootstrap,99% intervals for5contrasts. Report new72/old28/source breakdowns and source_group sensitivity. Prompt compression vs cached original global/free accuracy descriptive95%,not noninferiority proof. All fits/negative controls reported,no test-basedseed/model/thresholdselection.',
        timing='Six prechosenTRAIN screenshots additionally rerun original160-token global prompt;newshort global is independently generated on same image/hardware in same loaded runtime. Per-call sync/pixel/forward/token counts,PNG/candidatecanvas writes excluded. Shared-view workload sums are NOT standalone end-to-end latency. Historical MVP predictions same100 RGB/instruction authenticate;no new matched-cost MVPsuperiorityclaim.',
        outputs='Python logs,JSON,NPZ,TXT,staticPNG,noHTML;finish collection,training,evaluation,interpretation and audit.',
        hashes={str(p):sha(p) for p in frozen_inputs},backbone_inventory=read(ROOT/'runs/crossfit_candidates_v12/runtime_identity.json')['actual_model_files'])
    write(OUT/'protocol.json',spec)
    for p in SOURCES:
        q=OUT/'source_archive'/p.relative_to(ROOT);q.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,q)
    reg=read(ROOT/'CURRENT_RESEARCH.json');write(OUT/'registry_before.json',reg)
    reg['active_experiment']=dict(run_id=OUT.name,status='running',phase='registered',protocol_path=str(OUT/'protocol.json'),status_path=str(OUT/'status.json'),mainline_unchanged=True)
    write(ROOT/'CURRENT_RESEARCH.json',reg)
    status('registered',train=112,development=100,planned_generation_calls=854,planned_judge_prefills_max=848)
    return spec

class Runtime:
    def __init__(self):
        from clickagent.advantage_runtimes import configure_paths,RuntimeCounter
        configure_paths('coordinate')
        import torch
        from clickagent.grounded_composition_v28 import GroundedRuntime
        from clickagent.model_features import ExtractionConfig
        from clickagent.crossfit_proposer import load_crossfit_proposer
        torch.set_num_threads(4)
        cfg=dict(read(ROOT/'runs/crossfit_candidates_v12/runtime_identity.json')['runtime_contract'])
        cfg['layers']=tuple(cfg['layers']);cfg['conditions']=tuple(cfg['conditions']);cfg['offload_folder']=str(OUT/'offload')
        cfg['max_memory']={int(k) if str(k).isdigit() else k:v for k,v in cfg['max_memory'].items()}
        self.runtime=GroundedRuntime(ExtractionConfig(**cfg));self.torch=torch
        self.proposer=load_crossfit_proposer(ROOT/'runs/crossfit_candidates_v12/proposers/full.pt','cpu')[0].eval().requires_grad_(False)
        self.counter=RuntimeCounter();self.counter.attach(self.runtime.model,16)
        self.label_ids=[self.runtime.processor.tokenizer.encode(s,add_special_tokens=False) for s in ['A','B','C','D']]
        assert all(len(v)==1 for v in self.label_ids);self.label_ids=[v[0] for v in self.label_ids]
        write(OUT/'runtime.json',dict(metadata=self.runtime.runtime_metadata,label_ids=self.label_ids,
            devices={k:str(v) for k,v in self.runtime.model.hf_device_map.items()}))

    def generate(self,image,instruction,prompt,cap):
        before=self.counter.snapshot();a,m=self.runtime.infer(image,instruction,prompt,cap);after=self.counter.snapshot()
        m['counter']={k:after[k]-before[k] for k in before};return a,m

    def judge(self,image,instruction):
        from clickagent.model_features import _sync_cuda
        torch=self.torch;rt=self.runtime
        with rt._lock,torch.inference_mode():
            _sync_cuda();started=time.perf_counter();before=self.counter.snapshot()
            inputs,positions,grid=rt._prepare_image(image,instruction,JUDGE_PROMPT);states=[]
            def hook(module,args,output):states.append(output[0,-1].detach().float().cpu().numpy().copy())
            handle=rt.model.model.language_model.norm.register_forward_hook(hook)
            try:output=rt.model(**inputs,use_cache=False,logits_to_keep=1,output_attentions=False,output_hidden_states=False,return_dict=True)
            finally:handle.remove()
            logits=output.logits[0,-1,self.label_ids].float().cpu().numpy();del output
            assert len(states)==1 and states[0].shape==(4096,) and np.isfinite(states[0]).all()
            _sync_cuda();after=self.counter.snapshot()
            return states[0],dict(input_tokens=int(inputs['input_ids'].shape[-1]),elapsed_seconds=time.perf_counter()-started,
                logits=logits.tolist(),label_ids=self.label_ids,grid_shape=list(grid),generated_tokens=0,
                counter={k:after[k]-before[k] for k in before},full_instruction=instruction,system_prompt=JUDGE_PROMPT,
                feature='last normalized decoder token4096D,prefill of actual image+full instruction',ground_truth_in_model_input=False)

def collect_case(rt,c):
    from clickagent.grounded_composition_v28 import GLOBAL_PROMPT,LOCAL_PROMPT,pixel,fixed_crop
    from clickagent.crossfit_proposer import predict_raw
    assert sha(c['image'])==c['image_sha256']
    with Image.open(c['image']) as opened:image=opened.convert('RGB')
    try:
        size=list(image.size);assert size==c['image_size']
        gm_path=path(c,'raw/global')
        if not gm_path.exists():
            a,m=rt.generate(image,c['instruction'],SHORT_GLOBAL_PROMPT,64)
            raw,meta=proposal_observation(a,m);proposal=predict_raw(rt.proposer,raw,meta,size,64)
            gp=pixel(m['point'],size);center=first_valid(gp,proposal['point'],[size[0]//2,size[1]//2])
            first=fixed_crop(center,size);diverse,plan=plan_diverse(proposal['all_points'],proposal['all_scores'],size,first['box'])
            m.update(sample_id=c['sample_id'],global_point=gp,first_crop=first,diverse_crop=diverse,
                diverse_plan=plan,proposal=proposal,transform=None,image_sha256=c['image_sha256'],instruction=c['instruction'],labels_used=False)
            npz(gm_path,a,m);status('global_'+c['cohort'],sample_id=c['sample_id'],seconds=m['elapsed_seconds'],tokens=len(m['generated_token_ids']))
        _,gm=load(gm_path)
        for part in ['local0','refine','diverse']:
            p=path(c,'raw/'+part)
            if p.exists():continue
            if part=='local0':tr=gm['first_crop']
            elif part=='diverse':tr=gm['diverse_crop']
            else:
                _,fm=load(path(c,'raw/local0'));center=first_valid(fm['world_point'],gm['global_point'],gm['proposal']['point'],[size[0]//2,size[1]//2]);tr=fixed_crop(center,size)
            with image.crop(tr['box']).resize(tuple(tr['crop_size']),Image.Resampling.LANCZOS) as local:
                a,m=rt.generate(local,c['instruction'],LOCAL_PROMPT,56)
            m.update(sample_id=c['sample_id'],world_point=pixel(m['point'],m['image_size'],tr),transform=tr,image_sha256=c['image_sha256'],instruction=c['instruction'],labels_used=False)
            npz(p,{},m);del a
            status('local_'+c['cohort'],sample_id=c['sample_id'],part=part,seconds=m['elapsed_seconds'],tokens=len(m['generated_token_ids']))
        timing_ids=read(OUT/'data/prompt_timing.json')
        if c['sample_id'] in timing_ids and not path(c,'timing/full_global').exists():
            a,m=rt.generate(image,c['instruction'],GLOBAL_PROMPT,160);npz(path(c,'timing/full_global'),{},dict(m,sample_id=c['sample_id'],prompt_timing_only=True));del a
        fm=load(path(c,'raw/local0'))[1];dm=load(path(c,'raw/diverse'))[1];rm=load(path(c,'raw/refine'))[1]
        base=first_valid(fm['world_point'],gm['global_point'],gm['proposal']['point'])
        alternative=first_valid(dm['world_point'],gm['global_point'])
        eligible=eligible_pair(base,alternative)
        for blur in [False,True]:
            for reverse in [False,True]:
                part=f'judge/{"blur" if blur else "clear"}_{int(reverse)}';p=path(c,part)
                if p.exists():continue
                if not eligible:
                    npz(p,dict(hidden=np.zeros(4096,np.float32)),dict(sample_id=c['sample_id'],skipped=True,reason='no_distinct_valid_pair',ground_truth_in_model_input=False));continue
                left,right=(alternative,base) if reverse else (base,alternative)
                canvas,geometry=judge_canvas(image,left,right,blur)
                try:hidden,m=rt.judge(canvas,c['instruction'])
                finally:canvas.close()
                m.update(sample_id=c['sample_id'],base=base,alternative=alternative,reverse=reverse,blur=blur,geometry=geometry,skipped=False)
                npz(p,dict(hidden=hidden),m)
                status('judge_'+c['cohort'],sample_id=c['sample_id'],blur=blur,reverse=reverse,seconds=m['elapsed_seconds'])
        pair=dict(sample_id=c['sample_id'],cohort=c['cohort'],source=c['source'],group=c['group'],source_group=c['source_group'],
            base=base,alternative=alternative,global_point=gm['global_point'],local0=fm['world_point'],refine=rm['world_point'],diverse=dm['world_point'],
            refine_consensus=consensus([gm['global_point'],fm['world_point'],rm['world_point']],base),
            diverse_consensus=consensus([gm['global_point'],fm['world_point'],dm['world_point']],base),eligible=eligible,
            first_crop=gm['first_crop']['box'],diverse_crop=gm['diverse_crop']['box'],refine_crop=rm['transform']['box'],diverse_plan=gm['diverse_plan'],
            generation_seconds=sum(load(path(c,'raw/'+p))[1]['elapsed_seconds'] for p in ['global','local0','refine','diverse']),labels_used=False)
        clear=[load(path(c,f'judge/clear_{r}')) for r in [0,1]];blurred=[load(path(c,f'judge/blur_{r}')) for r in [0,1]]
        content=joint_vector(clear[0][0]['hidden'],clear[1][0]['hidden']);blur_vector=joint_vector(blurred[0][0]['hidden'],blurred[1][0]['hidden'])
        geo=pair_geometry(base,alternative,gm['global_point'],pair['first_crop'],pair['diverse_crop'],size,
            dict(base=fm['target'],alternative=dm['target'],global_=gm['target'])|{'global':gm['target']}) if eligible else np.zeros(7)
        def utility(m):
            if m['skipped']:return 0.
            logits=np.asarray(m['logits']);prob=np.exp(logits-logits.max());prob/=prob.sum();return float(prob[1]-prob[0])
        pair['zero_shot_utility']=.5*(utility(clear[0][1])-utility(clear[1][1]))
        pair['judge_seconds']=sum(m.get('elapsed_seconds',0.) for _,m in clear+blurred)
        costs={p:load(path(c,'raw/'+p))[1]['elapsed_seconds'] for p in ['global','local0','refine','diverse']}
        pair['workload_seconds']=dict(short_single=costs['global']+costs['local0'],
            refine_consensus=costs['global']+costs['local0']+costs['refine'],
            diverse_consensus=costs['global']+costs['local0']+costs['diverse'],
            content_policy=costs['global']+costs['local0']+costs['diverse']+sum(m.get('elapsed_seconds',0.) for _,m in clear))
        npz(path(c,'features'),dict(content=content,blurred=blur_vector,geometry=geo),pair)
    finally:image.close()

def records(cases,labels=None):
    from clickagent.public_evaluation import point_in_geometry
    rr=[]
    for c in cases:
        a,m=load(path(c,'features'));r=dict(m,**a)
        if labels is not None:
            def hit(p):return int(p is not None and point_in_geometry([p[0]/c['image_size'][0],p[1]/c['image_size'][1]],labels[c['sample_id']]))
            r.update(delta=hit(r['alternative'])-hit(r['base']),base_hit=hit(r['base']),alternative_hit=hit(r['alternative']))
        rr.append(r)
    return rr

def save_model(p,model):
    arrays={k:v for k,v in model.items() if isinstance(v,np.ndarray)}
    basis=model['basis'];arrays.update({'basis_'+k:v for k,v in basis.items()})
    meta={k:v for k,v in model.items() if k not in arrays and k!='basis'};npz(p,arrays,meta)
    return dict(path=str(p),sha256=sha(p),training_rgb=model['training_rgb'],repairs=model['repairs'],harms=model['harms'],ties=model['ties'])
def load_model(row):
    assert sha(row['path'])==row['sha256'];a,m=load(row['path']);m['basis']={k[6:]:v for k,v in a.items() if k.startswith('basis_')};m['alpha']=a['alpha'];return m

def train(cases):
    assert all(c['cohort']=='train' for c in cases)
    labels=read(OUT/'data/labels.json');rr=records(cases,labels);eligible=[r for r in rr if r['eligible']]
    assert eligible,'No valid distinct training pair; cannot fit content utility'
    folds={}
    for source in sorted({r['source'] for r in rr}):
        groups=sorted({r['source_group'] for r in rr if r['source']==source},key=lambda g:key(str(g)+'|v31_cv'))
        folds.update({(source,str(g)):j%3 for j,g in enumerate(groups)})
    cv=[];audits=[]
    splits=[('RGB_group',str(j),[r for r in eligible if folds[r['source'],str(r['source_group'])]!=j],[r for r in eligible if folds[r['source'],str(r['source_group'])]==j]) for j in range(3)]
    splits += [('source_LOSO',s,[r for r in eligible if r['source']!=s],[r for r in eligible if r['source']==s]) for s in sorted({r['source'] for r in eligible})]
    for kind,fold,fitting,validation in splits:
        if not fitting or not validation:continue
        assert not {r['group'] for r in fitting}&{r['group'] for r in validation}
        assert not {(r['source'],str(r['source_group'])) for r in fitting}&{(r['source'],str(r['source_group'])) for r in validation}
        audits.append(dict(kind=kind,fold=fold,fit_ids=[r['sample_id'] for r in fitting],validation_ids=[r['sample_id'] for r in validation]))
        for mode in MODES:
            model=fit(fitting,mode);pred=scores(model,validation)
            cv.extend(dict(kind=kind,fold=fold,mode=mode,sample_id=r['sample_id'],delta=r['delta'],predicted_utility=float(p),switch=bool(p>.1)) for r,p in zip(validation,pred))
        status('training_crossfit',kind=kind,fold=fold,fit_queries=len(fitting),validation_queries=len(validation))
    models={mode:save_model(OUT/'models'/f'{mode}.npz',fit(eligible,mode)) for mode in MODES}
    write(OUT/'training_crossfit.json',dict(rows=cv,splits=audits,eligible_queries=len(eligible),all_training_queries=len(rr),
        repairs=sum(r['delta']>0 for r in eligible),harms=sum(r['delta']<0 for r in eligible),ties=sum(r['delta']==0 for r in eligible)))
    inputs={str(path(c,'features')):sha(path(c,'features')) for c in cases}
    write(OUT/'frozen.json',dict(complete=True,frozen_at_utc=now(),models=models,development_used=False,training_ids=[c['sample_id'] for c in cases],training_feature_sha256=inputs))
    status('models_frozen',models=5,eligible_training_queries=len(eligible),repairs=sum(r['delta']>0 for r in eligible),harms=sum(r['delta']<0 for r in eligible))

def evaluate(cases):
    from clickagent.public_evaluation import point_in_geometry
    frozen=read(OUT/'frozen.json');assert frozen['complete'] and not frozen['development_used']
    labels=read(OUT/'data/labels.json');rr=records(cases);models={mode:load_model(row) for mode,row in frozen['models'].items()}
    eligible=[r for r in rr if r['eligible']];pred={mode:{r['sample_id']:float(p) for r,p in zip(eligible,scores(model,eligible))} for mode,model in models.items()}
    rows=[];baseline_index=read(OUT/'baseline_input_index.json');mvp_index=read(OUT/'mvp_input_index.json')
    for c,r in zip(cases,rr):
        def hit(p):return bool(p is not None and point_in_geometry([p[0]/c['image_size'][0],p[1]/c['image_size'][1]],labels[c['sample_id']]))
        old={}
        for part in ['global','free']:
            p=PRIOR/'raw'/part/(key(c['sample_id'])+'.npz');assert sha(p)==baseline_index[str(p)];old[part]=load(p)[1]
        from clickagent.grounded_composition_v28 import pixel
        methods=dict(short_single=r['base'],short_global=r['global_point'],short_local=r['local0'],
            refine_consensus=r['refine_consensus'],diverse_consensus=r['diverse_consensus'],
            original_global=pixel(old['global']['point'],old['global']['image_size'],old['global']['transform']),
            original_local=pixel(old['free']['point'],old['free']['image_size'],old['free']['transform']))
        mp=ROOT/'runs/mvp_repair_v21/mvp'/(key(c['sample_id'])+'.json');assert sha(mp)==mvp_index[str(mp)];methods['MVP_historical']=read(mp)['output']['point']
        switches={}
        for mode in MODES:
            utility=pred[mode].get(r['sample_id'],0.);methods[mode],reason=decide(r['base'],r['alternative'],utility,.1)
            methods[mode+'_zero'],_=decide(r['base'],r['alternative'],utility,0.)
            switches[mode]=dict(predicted_utility=utility,reason=reason,repair=bool(not hit(r['base']) and hit(methods[mode])),harm=bool(hit(r['base']) and not hit(methods[mode])))
        methods['zero_shot'],_=decide(r['base'],r['alternative'],r['zero_shot_utility'],.1)
        hits={name:hit(p) for name,p in methods.items()}
        hits['refine_union']=any(hit(p) for p in [r['global_point'],r['local0'],r['refine']])
        hits['diverse_union']=any(hit(p) for p in [r['global_point'],r['local0'],r['diverse']])
        def intersects(box):
            return any(min(box[2],b[2]*c['image_size'][0])>max(box[0],b[0]*c['image_size'][0]) and min(box[3],b[3]*c['image_size'][1])>max(box[1],b[1]*c['image_size'][1]) for b in labels[c['sample_id']]['boxes'])
        rows.append(dict(sample_id=c['sample_id'],source=c['source'],group=c['group'],source_group=c['source_group'],cohort=c['cohort'],methods=methods,hits=hits,switches=switches,eligible=r['eligible'],
            crops=dict(first_intersects=intersects(r['first_crop']),refine_intersects=intersects(r['refine_crop']),diverse_intersects=intersects(r['diverse_crop'])),
            diverse_plan=r['diverse_plan'],generation_seconds=r['generation_seconds'],judge_seconds=r['judge_seconds'],workload_seconds=r['workload_seconds']))
    write(OUT/'predictions.json',dict(rows=rows,models_frozen_at_utc=frozen['frozen_at_utc'],labels_used_in_inference=False))
    status('predictions_complete',queries=len(rows))

def bootstrap_weights(rows,field='group'):
    strata=defaultdict(lambda:defaultdict(list))
    for j,r in enumerate(rows):strata[r['source']][str(r[field])].append(j)
    rng=np.random.default_rng(20263110);w=np.zeros((5000,len(rows)))
    for groups in strata.values():
        groups=list(groups.values());draw=rng.integers(0,len(groups),(5000,len(groups)))
        for j,idx in enumerate(groups):w[:,idx]=(draw==j).sum(1)[:,None]
    return w/w.sum(1,keepdims=True)

def analyze():
    spec=verify();rows=read(OUT/'predictions.json')['rows'];result={};lines=[]
    for cohort in ['all','new_development','old_val']:
        rr=[r for r in rows if cohort=='all' or r['cohort']==cohort];w=bootstrap_weights(rr);wg=bootstrap_weights(rr,'source_group')
        names=list(rr[0]['hits']);summary={name:dict(percent=100*float(np.mean([r['hits'][name] for r in rr])),hits=sum(r['hits'][name] for r in rr)) for name in names}
        comparisons={}
        for a,b in spec['primary_contrasts']:
            diff=np.asarray([int(r['hits'][a])-int(r['hits'][b]) for r in rr],float);level=.99 if cohort=='all' else .95;alpha=(1-level)/2
            interval=(100*np.quantile(w@diff,[alpha,1-alpha])).tolist();sensitivity=(100*np.quantile(wg@diff,[alpha,1-alpha])).tolist()
            comparisons[a+'-'+b]=dict(delta_pp=100*float(diff.mean()),ci_pp=interval,source_group_ci_pp=sensitivity,level=level,positive=bool(interval[0]>0))
        replacements={mode:dict(switches=sum(r['methods'][mode]!=r['methods']['short_single'] for r in rr),
            repairs=sum(r['switches'][mode]['repair'] for r in rr),harms=sum(r['switches'][mode]['harm'] for r in rr)) for mode in MODES}
        result[cohort]=dict(n=len(rr),methods=summary,comparisons=comparisons,replacements=replacements,
            by_source={s:{name:100*float(np.mean([r['hits'][name] for r in rr if r['source']==s])) for name in names} for s in sorted({r['source'] for r in rr})},
            crops=dict(first_intersects=sum(r['crops']['first_intersects'] for r in rr),either_diverse_intersects=sum(r['crops']['first_intersects'] or r['crops']['diverse_intersects'] for r in rr),either_refine_intersects=sum(r['crops']['first_intersects'] or r['crops']['refine_intersects'] for r in rr)),
            workload_seconds={name:float(np.mean([r['workload_seconds'][name] for r in rr])) for name in rr[0]['workload_seconds']})
        lines += [f'{cohort}: n={len(rr)}']+[f'  {name}: {m["percent"]:.2f}% ({m["hits"]}/{len(rr)})' for name,m in summary.items()]
        lines += ['  contrasts: '+json.dumps(comparisons,ensure_ascii=False),'  repairs/harms: '+json.dumps(replacements,ensure_ascii=False)]
    cost=[]
    for sid in read(OUT/'data/prompt_timing.json'):
        c=next(c for c in read(OUT/'data/public.json') if c['sample_id']==sid)
        long=load(path(c,'timing/full_global'))[1];short=load(path(c,'raw/global'))[1]
        cost.append(dict(sample_id=sid,full_seconds=long['elapsed_seconds'],short_seconds=short['elapsed_seconds'],full_tokens=len(long['generated_token_ids']),short_tokens=len(short['generated_token_ids'])))
    timing={k:float(np.mean([r[k] for r in cost])) for k in ['full_seconds','short_seconds','full_tokens','short_tokens']}
    cv=read(OUT/'training_crossfit.json');cv_summary={}
    for kind in ['RGB_group','source_LOSO']:
        cv_summary[kind]={mode:dict(n=sum(r['kind']==kind and r['mode']==mode for r in cv['rows']),
            net_repairs=sum(r['delta'] for r in cv['rows'] if r['kind']==kind and r['mode']==mode and r['switch'])) for mode in MODES}
    result.update(prompt_timing=dict(n=6,per_case=cost,means=timing,note='Global stage ONLY,same6TRAIN images/hardware;not end-to-end method/MVP latency'),training_cv=cv_summary,
        training_pairs={k:cv[k] for k in ['eligible_queries','all_training_queries','repairs','harms','ties']},complete=True)
    write(OUT/'results.json',result)
    lines += ['', 'Prompt timing: '+json.dumps(result['prompt_timing'],ensure_ascii=False),
        'Training pair counts: '+json.dumps(result['training_pairs']), 'Training-only crossfit: '+json.dumps(cv_summary),
        'Development has been repeatedly observed. No independent MVP-superiority, end-to-end speed, or formal generalization claim. Shared evidence costs are not standalone timings. Mainline remains V15_3k.']
    (OUT/'readout.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    status('analysis_complete',primary_gates={k:v['positive'] for k,v in result['all']['comparisons'].items()})

def audit():
    spec=verify();cases=read(OUT/'data/public.json');frozen=read(OUT/'frozen.json');checked=0
    for p,h in frozen['training_feature_sha256'].items():assert sha(p)==h
    for row in frozen['models'].values():load_model(row)
    for c in cases:
        for part in ['raw/global','raw/local0','raw/refine','raw/diverse','judge/clear_0','judge/clear_1','judge/blur_0','judge/blur_1','features']:
            a,m=load(path(c,part));assert m['sample_id']==c['sample_id']
            if part.startswith('raw') or part.startswith('judge'):assert m['ground_truth_in_model_input'] is False
            checked+=1
    cv=read(OUT/'training_crossfit.json');lookup={c['sample_id']:c for c in cases}
    for split in cv['splits']:
        for f in ['group','source_group']:
            aa={(lookup[s]['source'],str(lookup[s][f])) for s in split['fit_ids']};bb={(lookup[s]['source'],str(lookup[s][f])) for s in split['validation_ids']};assert not aa&bb
    reg=read(ROOT/'CURRENT_RESEARCH.json');assert reg['run_id']=='quality_readout_v15_3k'
    result=read(OUT/'results.json');reg['active_experiment'].update(status='complete',phase='collected_trained_analyzed_verified',results_path=str(OUT/'readout.txt'),mainline_unchanged=True)
    write(ROOT/'CURRENT_RESEARCH.json',reg)
    write(OUT/'verification.json',dict(complete=True,at_utc=now(),case_artifacts_checked=checked,source_hashes_checked=len(spec['hashes']),crossfit_splits=len(cv['splits']),mainline_unchanged=True))
    write(OUT/'completion.json',dict(complete=True,at_utc=now(),protocol_sha256=sha(OUT/'protocol.json'),results_sha256=sha(OUT/'results.json'),verification_sha256=sha(OUT/'verification.json')))
    status('complete',queries=212,primary_gates={k:v['positive'] for k,v in result['all']['comparisons'].items()})

def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','all','analyze','audit']);args=p.parse_args()
    if args.stage=='prepare':prepare();return
    if args.stage=='analyze':analyze();return
    if args.stage=='audit':audit();return
    prepare();cases=read(OUT/'data/public.json');train_cases=[c for c in cases if c['cohort']=='train'];dev=[c for c in cases if c['cohort']!='train']
    status('loading_backbone');rt=Runtime()
    for i,c in enumerate(train_cases,1):
        if not path(c,'features').exists():collect_case(rt,c)
        status('training_observations',completed=i,total=len(train_cases),sample_id=c['sample_id'])
    if not (OUT/'frozen.json').exists():train(train_cases)
    assert read(OUT/'frozen.json')['complete']
    for i,c in enumerate(dev,1):
        if not path(c,'features').exists():collect_case(rt,c)
        status('development_observations',completed=i,total=len(dev),sample_id=c['sample_id'])
    evaluate(dev);analyze();audit()

if __name__=='__main__':
    try:main()
    except BaseException:
        status('failed',traceback=traceback.format_exc());raise
