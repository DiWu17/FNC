"""Fixed-pool representation x capacity experiment;local Python,no HTML."""
import os
for k,v in {'PYTHONDONTWRITEBYTECODE':'1','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','OPENBLAS_NUM_THREADS':'2','OMP_NUM_THREADS':'2','CUBLAS_WORKSPACE_CONFIG':':4096:8'}.items():os.environ.setdefault(k,v)
import argparse,gc,hashlib,json,shutil,subprocess,sys,time,traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import numpy as np
import torch
import torch.nn.functional as F
import run_bottleneck_v19 as prior
from clickagent.structure_readout_v22 import StructureReadout,spatial_sequence,projection,DIM
from clickagent.bottleneck_v19 import probe_loss

OUT=ROOT/'runs/structure_readout_v22';MAIN=prior.MAIN;SEEDS=prior.SEEDS;STEPS=2976
read,write,sha,save,key,now,digest=prior.read,prior.write,prior.sha,prior.save,prior.key,prior.now,prior.digest
SOURCES=[Path(__file__),ROOT/'src/clickagent/structure_readout_v22.py',ROOT/'tests/test_structure_readout_v22.py']
ARMS=['pooled32','ordered32','pooled64','ordered64']
def status(phase,worker=None,**fields):
    v=dict(phase=phase,complete=phase=='complete',updated_at_utc=now(),**fields)
    write(OUT/('status.json' if worker is None else f'workers/{worker}.status.json'),v);print(json.dumps(v,ensure_ascii=False),flush=True)
def np_save(p,a):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp')
    with tmp.open('wb') as f:np.save(f,a,allow_pickle=False)
    prior.replace(tmp,p)
def verify():
    v=read(OUT/'protocol.json')
    for p,h in v['hashes'].items():assert sha(p)==h,p
    for p,s in v['backbone_identity'].items():
        st=Path(p).stat();assert [st.st_size,st.st_mtime_ns]==s,p
    return v
def prepare():
    if (OUT/'protocol.json').exists():return verify()
    OUT.mkdir(parents=True,exist_ok=True)
    pre=read(OUT/'preflight.json');assert pre['complete']
    indices=read(prior.OUT/'data/indices.json');public=read(prior.OUT/'data/public.json')
    assert {k:len(v) for k,v in indices.items()}==dict(train=2973,old_val=172,new_development=300)
    write(OUT/'data/indices.json',indices);write(OUT/'data/public.json',public)
    identity=read(ROOT/'data/conditioned_v3/qwen_weight_identity.json');model=ROOT/'models/Qwen3-VL-8B-Instruct'
    prior_auth=read(ROOT/'runs/mvp_repair_v21/protocol.json')['backbone_files'];backbone={}
    for name,r in identity['files'].items():
        p=model/name;st=p.stat();old=prior_auth[str(p)]
        assert old['sha256']==r['sha256'] and st.st_size==old['bytes'] and st.st_mtime_ns==old['mtime_ns']
        backbone[str(p)]=[st.st_size,st.st_mtime_ns]
    inputs=SOURCES+[prior.OUT/'data/indices.json',prior.OUT/'data/public.json',prior.OUT/'normalization.pt',
        prior.OUT/'evaluation/train.json',prior.OUT/'evaluation/old_val.json',prior.OUT/'evaluation/new_development.json',
        MAIN/'models/frozen.json',ROOT/'src/clickagent/evidence_retention.py',ROOT/'src/clickagent/bottleneck_v19.py',
        ROOT/'scripts/run_bottleneck_v19.py',ROOT/'scripts/run_quality_readout_v15.py',OUT/'preflight.json',
        OUT/'data/indices.json',OUT/'data/public.json',ROOT/'data/conditioned_v3/qwen_weight_identity.json']
    adopted=read(MAIN/'models/frozen.json')['models']
    for s in [1,2,3]:inputs +=[Path(adopted[f'expanded_s{s}']['path']),prior.OUT/f'models/capacity_s{s}.pt']
    spec=dict(registered_at_utc=now(),hashes={str(p):sha(p) for p in inputs},backbone_identity=backbone,
        objective='Our complete candidate method:does retaining instruction/spatial structure help beyond an identical pooled readout,under matched parameter capacities?',
        cohort='Exact V19 train2973/2878nonempty,old172,new300 development. No V18fresh280 or new MVP inference.',
        immutable='Candidate coordinates,views,proposer,backbone,data,labels,2976steps,batch8,optimizer and3seeds fixed;no relabeling or tuning on development.',
        arms='Pooled32/ordered32/pooled64/ordered64;12newfits. Within each width same architecture,parameter count,initialization,batches,token counts,validity masks and forward shapes. Original3k and prior width64 capacity are archived references,not equal-architecture causal controls.',
        retained_inputs='Existing32base+294state remain. Add complete instruction causal text-only layer15 hidden sequence from same frozen8B,all tokens,no truncation,unit-normalized fixed64D Rademacher projection. Existing same-resolution cached3x3 fine vision cells+central coarse decoder image cell,each projected64D;relative spatial/view roles retained.',
        text_scope='Text-only contextual encoding of original instruction,not re-encoding image+instruction jointly. Does not retain original4096D states or generate new higher-resolution images. Projection seed20262201+d fixed,not learned. Identical instructions may share frozen text features.',
        pooled='Mean full-instruction token content repeated at every valid token position;mean valid3x3 fine-cell content repeated at valid fine positions. Coarse image token,roles,lengths,masks,view identities and all original32+294inputs unchanged. This is a matched structure ablation,not the exact legacy feature extractor.',
        architecture='Width32/64 original-style core plus GRU instruction encoder,query-conditioned spatial/view attention and zero-initialized extra score. All small readout weights trained from random initialization;no extra pretraining of the head;frozen8B adds inference compute.',
        optimizer=dict(steps=STEPS,seeds=list(SEEDS),lr=.0002,weight_decay=.02,batch_queries=8,gradient_clip=1.,loss='same multipositive/null listwise +.25balancedBCE',dropout=0),
        primary='New300:ordered32-pooled32,ordered64-pooled64,ordered32-original,structure-by-capacity interaction.5000 source-stratified RGB cluster bootstrap,98.75% Bonferroni4 intervals,fixed3-fit means. Practical positive threshold2pp,lower>0;interaction two-sided descriptive mechanism.',
        advancement='Ordered32 must improve both matched pooled32 and original by>=2pp with positive adjusted lower bounds;old172 ordered32-original point>=-2pp. No automatic adoption;any positive needs future independent data.',
        secondary='Train/old172 descriptive95%;new300 source-group sensitivity;per-source andconditional accuracy. Frozen ordered-model text-only,space-only,both pooling input interventions onnew300 are descriptive distribution shifts,not independent causal attribution.',
        interpretation='Ordered-pooled isolates joint retained text/spatial content at this frozen projected representation and head. New-vs-original also changes information,architecture and extraction cost;do not attribute it solely to representation. Negative result does not reject all richer/contextual/high-resolution readouts.',
        execution='Separate frozen8B text process thenCPU spatial extraction then2tracked GPU readout workers;all12checkpoints frozen before development evaluation.PythonJSONTXTPNG,noHTML.',
        parameter_counts={str(w):sum(p.numel() for p in StructureReadout(w).parameters()) for w in [32,64]},mainline_unchanged=True)
    for p in SOURCES:
        dest=OUT/'source_archive'/p.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest)
    write(OUT/'protocol.json',spec);reg=read(ROOT/'CURRENT_RESEARCH.json');write(OUT/'registry_before.json',reg)
    reg['active_experiment']=dict(run_id=OUT.name,status='registered',phase='protocol_registered',protocol_path=str(OUT/'protocol.json'),status_path=str(OUT/'status.json'),mainline_unchanged=True,new_fits=12)
    write(ROOT/'CURRENT_RESEARCH.json',reg);write(ROOT/'research/decisions/20261108_structure_readout_v22.json',dict(user_authorization='做一下这个对照试验吧',protocol_sha256=sha(OUT/'protocol.json'),mainline_unchanged=True));status('registered',new_fits=12)
    return spec

def text_features():
    verify();path=OUT/'text/index.json'
    if path.exists():assert read(path)['complete'];return
    from transformers import AutoTokenizer,Qwen3VLForConditionalGeneration
    tokenizer=AutoTokenizer.from_pretrained(ROOT/'models/Qwen3-VL-8B-Instruct',local_files_only=True)
    entries=read(OUT/'data/indices.json');public=read(OUT/'data/public.json');unique={}
    for e in [e for co in entries.values() for e in co if e['n']]:
        text=public[e['sample_id']]['instruction'];unique[hashlib.sha256(text.encode()).hexdigest()]=text
    todo=[];records={}
    for h,text in sorted(unique.items()):
        receipt=OUT/f'text/cases/{h}.json'
        if receipt.exists():
            r=read(receipt);assert sha(r['path'])==r['sha256'] and r['instruction_sha256']==h;records[h]=r
        else:todo.append((h,text,tokenizer(text,add_special_tokens=False)['input_ids']))
    if todo:
        status('loading_frozen_text_backbone',completed=len(records),total=len(unique))
        model=Qwen3VLForConditionalGeneration.from_pretrained(ROOT/'models/Qwen3-VL-8B-Instruct',torch_dtype=torch.float16,device_map='balanced',max_memory={0:'12GiB',1:'12GiB'},attn_implementation='sdpa',local_files_only=True).eval().requires_grad_(False)
        device=model.get_input_embeddings().weight.device;captured=[]
        def hook(module,args,output):
            hidden=output[0] if isinstance(output,(tuple,list)) else output
            values=hidden.float();values=values/values.norm(dim=-1,keepdim=True).clamp_min(1e-8)
            p=torch.as_tensor(projection(4096),device=values.device)
            captured.append((values@p).half().cpu().numpy())
        handle=model.model.language_model.layers[14].register_forward_hook(hook)
        # Public synthetic runtime check before any dataset feature is accepted.
        smoke_ids=tokenizer('Click the blue save button.',add_special_tokens=False)['input_ids'];captured.clear()
        with torch.inference_mode():smoke_output=model(input_ids=torch.tensor([smoke_ids],device=device),attention_mask=torch.ones(1,len(smoke_ids),dtype=torch.long,device=device),use_cache=False,logits_to_keep=1,return_dict=True)
        del smoke_output;assert len(captured)==1 and captured[0].shape==(1,len(smoke_ids),DIM) and np.isfinite(captured[0]).all()
        write(OUT/'text/runtime_smoke.json',dict(complete=True,synthetic=True,all_tokens=len(smoke_ids),shape=list(captured[0].shape),frozen_backbone=True,no_labels=True))
        todo.sort(key=lambda r:(len(r[2]),r[0]));started=time.perf_counter()
        for start in range(0,len(todo),8):
            batch=todo[start:start+8];lengths=[len(r[2]) for r in batch];assert min(lengths)>0
            ids=torch.full((len(batch),max(lengths)),tokenizer.pad_token_id or 151643,dtype=torch.long,device=device);mask=torch.zeros_like(ids)
            for j,(_,_,tokens) in enumerate(batch):ids[j,:len(tokens)]=torch.tensor(tokens,device=device);mask[j,:len(tokens)]=1
            captured.clear()
            with torch.inference_mode():output=model(input_ids=ids,attention_mask=mask,use_cache=False,logits_to_keep=1,return_dict=True)
            del output;assert len(captured)==1
            for j,(h,text,tokens) in enumerate(batch):
                v=captured[0][j,:len(tokens)].copy();assert v.shape==(len(tokens),DIM) and np.isfinite(v).all()
                dest=OUT/f'text/arrays/{h}.npy';np_save(dest,v)
                r=dict(path=str(dest),sha256=sha(dest),instruction_sha256=h,n_tokens=len(tokens),token_ids_sha256=digest(tokens),truncated=False,ground_truth_used=False)
                write(OUT/f'text/cases/{h}.json',r);records[h]=r
            if start%40==0 or start+8>=len(todo):status('extracting_full_instruction_text',completed=len(records),total=len(unique),elapsed_seconds=time.perf_counter()-started,max_tokens_in_batch=max(lengths))
        handle.remove();del model;gc.collect();torch.cuda.empty_cache()
    write(path,dict(complete=True,unique_instructions=len(records),records=records,protocol_sha256=sha(OUT/'protocol.json'),input='plain complete original instruction,text-only causal layer15,all tokens'))
    status('text_complete',unique_instructions=len(records))

def spatial_one(task):
    e,public=task;receipt=OUT/f'spatial/cases/{key(e["sample_id"])}.json'
    if receipt.exists():
        r=read(receipt)
        for name in ['features','mask']:assert sha(r[name+'_path'])==r[name+'_sha256']
        return r
    assert sha(e['path'])==e['sha256'] and sha(e['pipeline_path'])==e['pipeline_sha256']
    with np.load(e['path'],allow_pickle=False) as z:points=z['points'].copy()
    features=np.zeros((len(points),4,10,DIM),np.float16);mask=np.zeros((len(points),4,10),bool)
    pipe=read(e['pipeline_path']);assert not pipe['ground_truth_in_model_inputs'];raw={}
    views=[dict(feature_path=pipe['global_feature_path'],feature_sha256=pipe['global_feature_sha256'],transform=None)]+pipe['local_views']
    if len(points):
        for j,v in enumerate(views):
            assert sha(v['feature_path'])==v['feature_sha256'];raw[v['feature_path']]=v['feature_sha256']
            with np.load(v['feature_path'],allow_pickle=False) as z:
                meta=json.loads(str(z['metadata']));arrays={k:z[k].copy() for k in ['image_hidden','vision_hidden']}
            prior.authenticate_meta(meta,public,legacy_global=j==0 and 'runtime_identity_sha256' in pipe)
            features[:,j],mask[:,j]=spatial_sequence(arrays,meta,points,public['image_size'],v.get('transform'))
    fp=OUT/f'spatial/arrays/{key(e["sample_id"])}.npy';mp=fp.with_suffix('.mask.npy');np_save(fp,features);np_save(mp,mask)
    r=dict(sample_id=e['sample_id'],features_path=str(fp),features_sha256=sha(fp),mask_path=str(mp),mask_sha256=sha(mp),base_sha256=e['sha256'],raw_sha256=raw,ground_truth_used=False)
    write(receipt,r);return r
def spatial_features():
    verify();path=OUT/'spatial/index.json'
    if path.exists():assert read(path)['complete'];return
    indices=read(OUT/'data/indices.json');public=read(OUT/'data/public.json');all_e=[e for co in indices.values() for e in co];result=[];started=time.perf_counter()
    with ThreadPoolExecutor(max_workers=2) as pool:
        for j,r in enumerate(pool.map(spatial_one,[(e,public[e['sample_id']]) for e in all_e]),1):
            result.append(r)
            if j%25==0 or j==len(all_e):status('extracting_cached_spatial_sequences',completed=j,total=len(all_e),elapsed_seconds=time.perf_counter()-started)
    write(path,dict(complete=True,cases=result,protocol_sha256=sha(OUT/'protocol.json')))

class Cache:
    def __init__(self,cohort,device):
        self.device=device;self.records=prior.legacy.load_records(read(OUT/'data/indices.json')[cohort]);self.data=[]
        norm=torch.load(prior.OUT/'normalization.pt',weights_only=True,map_location='cpu');self.norm=norm
        spatial={e['sample_id']:e for e in read(OUT/'spatial/index.json')['cases']};text=read(OUT/'text/index.json')['records'];public=read(OUT/'data/public.json');texts={}
        for r in self.records:
            e=spatial[r['sample_id']];assert e['base_sha256']==r['sha256']
            for name in ['features','mask']:assert sha(e[name+'_path'])==e[name+'_sha256']
            h=hashlib.sha256(public[r['sample_id']]['instruction'].encode()).hexdigest()
            if h not in texts:
                if h in text:
                    t=text[h];assert sha(t['path'])==t['sha256'];values=np.load(t['path'],allow_pickle=False)
                else:
                    assert not r['n'],'Nonempty query must have full instruction features'
                    values=np.zeros((1,DIM),np.float16)
                texts[h]=torch.tensor(values,device=device)
            self.data.append(dict(base=torch.tensor((r['base']-norm['mean'].numpy())/norm['std'].numpy(),device=device),
                state=torch.from_numpy(np.array(r['state'])),mask=torch.from_numpy(np.array(r['mask'])),
                spatial=torch.tensor(np.load(e['features_path'],allow_pickle=False),device=device),
                spatial_mask=torch.tensor(np.load(e['mask_path'],allow_pickle=False),device=device),text=texts[h],labels=torch.tensor(r['labels'],device=device)))
    def batch(self,indices):
        rows=[self.data[int(i)] for i in indices];lengths=[len(r['base']) for r in rows];tmax=max(r['state'].shape[2] for r in rows);lmax=max(len(r['text']) for r in rows)
        args=dict(base=torch.cat([r['base'] for r in rows]),
            state=torch.cat([F.pad(r['state'],(0,0,0,tmax-r['state'].shape[2])) for r in rows]).to(self.device),
            mask=torch.cat([F.pad(r['mask'],(0,tmax-r['mask'].shape[2])) for r in rows]).to(self.device),
            spatial=torch.cat([r['spatial'] for r in rows]),spatial_mask=torch.cat([r['spatial_mask'] for r in rows]),
            text=torch.stack([F.pad(r['text'],(0,0,0,lmax-len(r['text']))) for r in rows]),
            text_mask=torch.stack([torch.arange(lmax,device=self.device)<len(r['text']) for r in rows]),
            candidate_case=torch.repeat_interleave(torch.arange(len(rows),device=self.device),torch.tensor(lengths,device=self.device)))
        return args,lengths,[r['labels'] for r in rows]

def fit(arm,seed,cache,worker):
    name=f'{arm}_s{SEEDS.index(seed)+1}';receipt=OUT/f'models/{name}.json'
    if receipt.exists():assert sha(read(receipt)['checkpoint_path'])==read(receipt)['checkpoint_sha256'];return
    width=32 if arm.endswith('32') else 64;condition=arm[:-2];torch.manual_seed(seed);np.random.seed(seed)
    model=StructureReadout(width).to(cache.device);initial=prior.state_identity(model);optimizer=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=.02)
    identity=dict(protocol_sha256=sha(OUT/'protocol.json'),spatial_index_sha256=sha(OUT/'spatial/index.json'),text_index_sha256=sha(OUT/'text/index.json'),arm=arm,seed=seed)
    resume=OUT/f'models/{name}.resume.pt';step=0;epoch=1;position=0;elapsed=0.;history=[]
    if resume.exists():
        old=torch.load(resume,weights_only=True,map_location='cpu');assert old['identity']==identity
        model.load_state_dict(old['state_dict']);optimizer.load_state_dict(old['optimizer']);step,epoch,position,elapsed,history=[old[k] for k in ['step','epoch','position','elapsed','history']]
        torch.set_rng_state(old['torch_rng']);torch.cuda.set_rng_state(old['cuda_rng'],cache.device)
    fitting=[i for i,r in enumerate(cache.records) if r['n']];started=time.perf_counter();losses=[];model.train()
    status('training',worker=worker,model=name,step=step,total_steps=STEPS)
    while step<STEPS:
        ix,epoch,position=prior.legacy.next_batch(len(fitting),seed,epoch,position);args,lengths,labels=cache.batch([fitting[int(i)] for i in ix]);optimizer.zero_grad(set_to_none=True)
        score=model(**args,condition=condition);loss=torch.stack([probe_loss(z,y,model.null,.25) for z,y in zip(score.split(lengths),labels)]).mean()
        assert torch.isfinite(loss);loss.backward();assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();step+=1;losses.append(float(loss.detach()))
        if step%100==0 or step==STEPS:
            h=dict(step=step,epoch=epoch,position=position,train_loss=float(np.mean(losses)),elapsed_seconds=elapsed+time.perf_counter()-started);history.append(h);losses=[]
            status('training',worker=worker,model=name,total_steps=STEPS,**h)
        if step%250==0 or step==STEPS:save(resume,dict(identity=identity,state_dict=model.state_dict(),optimizer=optimizer.state_dict(),step=step,epoch=epoch,position=position,elapsed=elapsed+time.perf_counter()-started,history=history,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(cache.device)))
    cp=OUT/f'models/{name}.pt';save(cp,dict(state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},identity=identity,width=width,condition=condition))
    write(receipt,dict(complete=True,arm=arm,name=name,seed=seed,steps=STEPS,parameter_count=sum(p.numel() for p in model.parameters()),initial_state_sha256=initial,checkpoint_path=str(cp),checkpoint_sha256=sha(cp),fitting_queries=len(fitting),manifest_queries=len(cache.records),history=history,validation_used_for_selection=False))
    del model,optimizer;gc.collect();torch.cuda.empty_cache()
def worker_run(worker):
    verify();torch.cuda.set_device(worker);torch.set_num_threads(2);device=f'cuda:{worker}';status('loading_train',worker=worker)
    cache=Cache('train',device);width=32 if worker==0 else 64
    for seed in SEEDS:
        for condition in ['pooled','ordered']:fit(f'{condition}{width}',seed,cache,worker)
    status('complete',worker=worker)
def child(stage):
    stdout=OUT/f'{stage}.stdout.log';stderr=OUT/f'{stage}.stderr.log'
    with stdout.open('a',encoding='utf-8') as so,stderr.open('a',encoding='utf-8') as se:p=subprocess.Popen([sys.executable,'-B','-X','utf8','-u',str(Path(__file__).resolve()),'--stage',stage],cwd=ROOT,stdout=so,stderr=se)
    write(OUT/f'{stage}.execution.json',dict(pid=p.pid,started_at_utc=now()))
    while p.poll() is None:time.sleep(45)
    assert p.returncode==0,f'{stage} exit{p.returncode};inspect {stderr}'
def train_all():
    verify();processes=[]
    for worker in [0,1]:
        p=OUT/f'workers/{worker}';p.parent.mkdir(parents=True,exist_ok=True)
        with Path(str(p)+'.stdout.log').open('a',encoding='utf-8') as so,Path(str(p)+'.stderr.log').open('a',encoding='utf-8') as se:
            process=subprocess.Popen([sys.executable,'-B','-X','utf8','-u',str(Path(__file__).resolve()),'--stage','worker','--worker',str(worker)],cwd=ROOT,stdout=so,stderr=se)
        processes.append(process)
    write(OUT/'workers/execution.json',dict(pids=[p.pid for p in processes],started_at_utc=now()))
    while True:
        codes=[p.poll() for p in processes];progress={str(i):read(OUT/f'workers/{i}.status.json') if (OUT/f'workers/{i}.status.json').exists() else {} for i in [0,1]}
        status('training_workers',exit_codes=codes,workers=progress)
        if any(c is not None and c!=0 for c in codes):raise RuntimeError('Tracked training worker failed;do not duplicate active sibling')
        if all(c==0 for c in codes):break
        time.sleep(45)
    registry={}
    for s in [1,2,3]:
        for arm in ARMS:
            name=f'{arm}_s{s}';r=read(OUT/f'models/{name}.json');assert r['complete'] and r['steps']==STEPS and sha(r['checkpoint_path'])==r['checkpoint_sha256'];registry[name]=r
        for w in [32,64]:assert registry[f'pooled{w}_s{s}']['initial_state_sha256']==registry[f'ordered{w}_s{s}']['initial_state_sha256']
    write(OUT/'models/frozen.json',dict(complete=True,frozen_at_utc=now(),models=registry,protocol_sha256=sha(OUT/'protocol.json')));status('models_frozen',new_fits=12)

@torch.inference_mode()
def evaluate():
    verify();registry=read(OUT/'models/frozen.json');assert registry['complete'];device='cuda:0';torch.cuda.set_device(0);models={}
    for name,r in registry['models'].items():
        assert sha(r['checkpoint_path'])==r['checkpoint_sha256'];v=torch.load(r['checkpoint_path'],weights_only=True,map_location='cpu');m=StructureReadout(v['width']).to(device).eval();m.load_state_dict(v['state_dict']);models[name]=(m,v['condition'])
    for co in ['train','old_val','new_development']:
        dest=OUT/f'evaluation/{co}.json'
        if dest.exists():assert read(dest)['complete'];continue
        status('loading_evaluation',cohort=co);cache=Cache(co,device);vectors={n:[] for n in models};losses={n:[] for n in models};flats={n:[] for n in models};interventions={f'{n}_{a}':[] for n in models if n.startswith('ordered') for a in ['text_pooled','space_pooled','both_pooled']} if co=='new_development' else {}
        for start in range(0,len(cache.records),8):
            ids=list(range(start,min(start+8,len(cache.records))));args,lengths,labels=cache.batch(ids)
            for name,(model,condition) in models.items():
                chunks=model(**args,condition=condition).split(lengths)
                for z,y in zip(chunks,labels):
                    vectors[name].append(bool(len(z) and y[int(z.argmax())]));flats[name].append(z.float().cpu().numpy());
                    if len(z):losses[name].append(float(probe_loss(z,y,model.null,.25)))
                if co=='new_development' and name.startswith('ordered'):
                    for ab in ['text_pooled','space_pooled','both_pooled']:
                        for z,y in zip(model(**args,condition=condition,ablation=ab).split(lengths),labels):interventions[f'{name}_{ab}'].append(bool(len(z) and y[int(z.argmax())]))
            if start%80==0 or start+8>=len(cache.records):status('evaluating',cohort=co,completed=min(start+8,len(cache.records)),total=len(cache.records))
        reference=read(prior.OUT/f'evaluation/{co}.json');assert reference['sample_ids']==[r['sample_id'] for r in cache.records]
        for name,v in reference['hit_vectors'].items():
            if name.startswith('original') or name.startswith('capacity'):vectors[name]=v
        pred=OUT/f'evaluation/{co}.scores.npz';pred.parent.mkdir(exist_ok=True,parents=True)
        with pred.open('wb') as f:np.savez_compressed(f,offsets=np.r_[0,np.cumsum([r['n'] for r in cache.records])],**{n:np.concatenate(v) for n,v in flats.items()})
        write(dest,dict(complete=True,cohort=co,n=len(cache.records),hit_vectors=vectors,intervention_hit_vectors=interventions,common_loss={n:float(np.mean(v)) for n,v in losses.items()},oracle_hits=sum(bool(np.any(r['labels'])) for r in cache.records),sample_ids=reference['sample_ids'],scores_path=str(pred),scores_sha256=sha(pred),model_freeze_sha256=sha(OUT/'models/frozen.json')))
        del cache,args;gc.collect();torch.cuda.empty_cache()

def bootstrap(values,entries,level,group='group'):
    strata={}
    for i,e in enumerate(entries):strata.setdefault(e['source'],{}).setdefault(e[group],[]).append(i)
    clusters=[list(v.values()) for v in strata.values()];rng=np.random.default_rng(20262202)
    diffs={'structure_small':values['ordered32']-values['pooled32'],'structure_large':values['ordered64']-values['pooled64'],'net_small':values['ordered32']-values['original'],'capacity_interaction':values['ordered64']-values['pooled64']-values['ordered32']+values['pooled32']}
    draws={k:[] for k in diffs}
    for _ in range(5000):
        ix=[i for groups in clusters for j in rng.integers(0,len(groups),len(groups)) for i in groups[j]]
        for k,v in diffs.items():draws[k].append(v[:,ix].mean())
    alpha=1-level
    return {k:dict(delta_pp=100*float(v.mean()),seed_deltas_pp=(100*v.mean(1)).tolist(),ci_pp=(100*np.quantile(draws[k],[alpha/2,1-alpha/2])).tolist(),interval_percent=level*100) for k,v in diffs.items()}
def analyze():
    verify();indices=read(OUT/'data/indices.json');results={};lines=['V22 自有完整候选：保留文本/空间结构 × 读出容量。','原2973训练清单，2878非空训练，固定2976步；12拟合先冻结，再评估。','pooled和ordered都增加相同的文本/视觉内容；它们比较的是结构保留。原版是额外参考，不能把相对原版的变化只归因于结构。']
    for co in ['train','old_val','new_development']:
        r=read(OUT/f'evaluation/{co}.json');values={a:np.asarray([r['hit_vectors'][f'{a}_s{s}'] for s in [1,2,3]],float) for a in ['original','capacity']+ARMS}
        methods={a:dict(accuracy_percent=100*float(v.mean()),seed_hits=v.sum(1).astype(int).tolist(),conditional_accuracy_percent=100*float(v.sum(1).mean())/r['oracle_hits'],by_source={source:100*float(v[:,[i for i,e in enumerate(indices[co]) if e['source']==source]].mean()) for source in sorted({e['source'] for e in indices[co]})}) for a,v in values.items()}
        comp=bootstrap(values,indices[co],.9875 if co=='new_development' else .95);results[co]=dict(n=r['n'],oracle_hits=r['oracle_hits'],methods=methods,comparisons=comp)
        if co=='new_development':
            results[co]['source_group_sensitivity95']=bootstrap(values,indices[co],.95,'source_group')
            results[co]['input_interventions']={f'{a}_{ab}':dict(accuracy_percent=100*float(np.asarray([r['intervention_hit_vectors'][f'{a}_s{s}_{ab}'] for s in [1,2,3]],float).mean()),seed_hits=[sum(r['intervention_hit_vectors'][f'{a}_s{s}_{ab}']) for s in [1,2,3]]) for a in ['ordered32','ordered64'] for ab in ['text_pooled','space_pooled','both_pooled']}
        lines+=['',f'{co},n={r["n"]},候选可达={r["oracle_hits"]}']
        for a,m in methods.items():lines.append(f'{a}: {m["accuracy_percent"]:.2f}%,三种子{m["seed_hits"]},可达条件{m["conditional_accuracy_percent"]:.2f}%')
        for a,c in comp.items():lines.append(f'{a}: {c["delta_pp"]:+.2f}pp,{c["interval_percent"]:.2f}%区间{c["ci_pp"]}')
    c=results['new_development']['comparisons'];gates={k:v['delta_pp']>=2 and v['ci_pp'][0]>0 for k,v in c.items() if k!='capacity_interaction'}
    advance=bool(gates['structure_small'] and gates['net_small'] and results['old_val']['comparisons']['net_small']['delta_pp']>=-2)
    lines+=['','登记正向门槛：'+str(gates),'进入后续独立验证候选门槛：'+str(advance),'新开发单独池化文本/空间的冻结输入干预：'+str(results['new_development']['input_interventions']),
        '所有输入都保留原32+294特征。序列条件只改变新增分支内容的保留方式，不能把负结果当作原始特征已足够、或全部信息压缩均无害的证明。',
        '完整指令是独立文本编码，并非图像条件下的全部指令token隐状态；64维投影、原图缓存分辨率和该读出结构均限制结论。',
        '开发集历史复用；未读取原fresh280，未训练底座，未自动采用，未生成HTML。']
    write(OUT/'results.json',dict(complete=True,cohorts=results,primary_positive_gates=gates,advance_to_independent_test=advance,mainline_unchanged=True,independent_confirmation=False))
    (OUT/'results.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names=['original','capacity']+ARMS;x=np.arange(len(names));fig,axes=plt.subplots(1,2,figsize=(12,4.3))
    for offset,co,label in [(-.18,'new_development','New dev300'),(.18,'old_val','Old dev172')]:axes[0].bar(x+offset,[results[co]['methods'][a]['accuracy_percent'] for a in names],.36,label=label)
    axes[0].set_xticks(x,names,rotation=25);axes[0].set_ylim(0,100);axes[0].set_ylabel('Accuracy (%)');axes[0].legend(fontsize=8)
    for j,k in enumerate(['structure_small','structure_large','net_small','capacity_interaction']):
        v=c[k];axes[1].plot(v['ci_pp'],[j,j],color='#2878B5');axes[1].plot(v['delta_pp'],j,'o',color='#2878B5')
    axes[1].set_yticks(range(4),['Structure width32','Structure width64','Ordered32 - original','Capacity interaction']);axes[1].axvline(0,color='gray');axes[1].set_xlabel('New dev delta pp;adjusted98.75% CI');fig.suptitle('V22 reused-development representation study');fig.tight_layout();fig.savefig(OUT/'summary.png',dpi=170);plt.close(fig)
    verify();files=[OUT/p for p in ['protocol.json','text/index.json','spatial/index.json','models/frozen.json','results.json','results.txt','summary.png']]+[OUT/f'evaluation/{co}.json' for co in indices]
    write(OUT/'completion.json',dict(complete=True,completed_at_utc=now(),artifact_sha256={str(p):sha(p) for p in files},mainline_unchanged=True))
    reg=read(ROOT/'CURRENT_RESEARCH.json');reg['active_experiment'].update(status='complete',phase='training_evaluation_analysis_complete',results_path=str(OUT/'results.txt'),completion_path=str(OUT/'completion.json'),primary_positive_gates=gates,advance_to_independent_test=advance);reg['completed_experiments'].append(dict(run_id=OUT.name,status='complete',analysis_path=str(OUT/'results.txt'),mainline_unchanged=True));write(ROOT/'CURRENT_RESEARCH.json',reg)
    status('complete',primary_positive_gates=gates,advance_to_independent_test=advance);print('\n'.join(lines),flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['prepare','text','spatial','worker','train','evaluate','analyze','all'],default='all');p.add_argument('--worker',type=int,choices=[0,1]);a=p.parse_args();torch.set_num_threads(2)
    try:
        prepare()
        if a.stage=='text':text_features()
        if a.stage=='spatial':spatial_features()
        if a.stage=='worker':worker_run(a.worker)
        if a.stage in ['all','train']:
            if a.stage=='all':child('text');child('spatial')
            train_all()
        if a.stage in ['all','evaluate']:evaluate()
        if a.stage in ['all','analyze']:analyze()
    except BaseException:
        status('failed',worker=a.worker if a.stage=='worker' else None,error=traceback.format_exc());raise
if __name__=='__main__':main()
