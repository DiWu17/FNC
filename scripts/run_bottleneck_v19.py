"""Registered fixed-pool bottleneck probes; Python/JSON/TXT, no HTML or VLM calls."""
import os
os.environ.setdefault('PYTHONDONTWRITEBYTECODE', '1')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
os.environ.setdefault('OMP_NUM_THREADS', '2')
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts')]
import numpy as np
import torch
import torch.nn.functional as F
from clickagent.bottleneck_v19 import BottleneckModel, ROI_DIM, local_view_features, shuffle_indices, probe_loss
from clickagent.evidence_retention import RetentionModel
import run_quality_readout_v15 as legacy

OUT = ROOT/'runs/bottleneck_v19'
MAIN = ROOT/'runs/quality_readout_v15_3k'
FULL = ROOT/'runs/full_scaling_v18'
SEEDS = (20261311, 20261312, 20261313)
STEPS = 2976
ARMS = ['capacity', 'visual', 'visual_shuffled', 'listwise_only']
CONTRASTS = {'capacity': ('capacity', 'original'), 'visual_vs_capacity': ('visual', 'capacity'),
             'visual_alignment': ('visual', 'visual_shuffled'), 'loss_objective': ('listwise_only', 'capacity')}
SOURCES = [Path(__file__), ROOT/'src/clickagent/bottleneck_v19.py', ROOT/'tests/test_bottleneck_v19.py',
           ROOT/'src/clickagent/evidence_retention.py', ROOT/'scripts/run_quality_readout_v15.py']


def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def now(): return datetime.now(timezone.utc).isoformat()
def key(s): return hashlib.sha256(s.encode()).hexdigest()[:24]
def digest(v): return hashlib.sha256(json.dumps(v, sort_keys=True).encode()).hexdigest()


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024), b''): h.update(block)
    return h.hexdigest()


def replace(a, b):
    for attempt in range(40):
        try: os.replace(a,b); return
        except PermissionError as e:
            if getattr(e,'winerror',None) not in (5,32,33) or attempt==39: raise
            time.sleep(.1)


def write(p, value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');replace(tmp,p)


def save(p, value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.tmp');torch.save(value,tmp);replace(tmp,p)


def status(phase, worker=None, **fields):
    row=dict(phase=phase,complete=phase=='complete',updated_at_utc=now(),pid=os.getpid(),**fields)
    path=OUT/'status.json' if worker is None else OUT/f'workers/{worker}.status.json'
    write(path,row);print(json.dumps(row,ensure_ascii=False),flush=True)


def verify():
    spec=read(OUT/'protocol.json')
    for p,h in spec['hashes'].items():assert sha(p)==h,p
    return spec


def authenticate_meta(meta, public, legacy_global=False):
    assert meta['sample_id']==public['sample_id']
    assert meta['image_sha256']==public['image_sha256']
    if 'instruction_sha256' in meta:
        assert meta['instruction_sha256']==hashlib.sha256(public['instruction'].encode()).hexdigest()
    else:
        # Audited V13 global caches predate this field; local/new caches have it.
        # Caller has already verified the sealed pipeline and raw byte hashes.
        assert legacy_global and not public['sample_id'].startswith('expansion15:')


def prepare():
    if (OUT/'protocol.json').exists():return verify()
    entries=read(MAIN/'data/old_train_index.json')+read(MAIN/'features/new_train/index.json')['cases']
    dev=read(FULL/'data/development_indices.json')
    assert len(entries)==2973 and len(dev['old_val'])==172 and len(dev['new_development'])==300
    indices=dict(train=entries,**dev)
    train_rgb={e['group'] for e in entries}
    assert not train_rgb & {e['group'] for c in dev.values() for e in c}
    cases={}
    for path in [ROOT/'runs/multiregion_v13/data/train.jsonl',ROOT/'runs/multiregion_v13/data/val.jsonl',
                 MAIN/'data/new_train.jsonl',MAIN/'data/new_reserve.jsonl']:
        for line in path.read_text(encoding='utf-8').splitlines():
            c=json.loads(line)
            cases[c['sample_id']]={k:c[k] for k in ['sample_id','image_size','image_sha256','instruction']}
    assert all(e['sample_id'] in cases for cohort in indices.values() for e in cohort)
    write(OUT/'data/indices.json',indices);write(OUT/'data/public.json',cases)
    inputs=SOURCES+[MAIN/'data/old_train_index.json',MAIN/'features/new_train/index.json',
                    FULL/'data/development_indices.json',MAIN/'models/frozen.json',
                    OUT/'data/indices.json',OUT/'data/public.json',
                    ROOT/'runs/multiregion_v13/data/train.jsonl',ROOT/'runs/multiregion_v13/data/val.jsonl',
                    ROOT/'runs/multiregion_v13/final_audit.json']
    spec=dict(registered_at_utc=now(),hashes={str(p):sha(p) for p in inputs},seeds=list(SEEDS),steps=STEPS,
        fixed='Same2973 training manifest, actual complete candidate pool, frozen original coordinates/views/proposer/backbone, all3 seeds and final2976 updates. Empty training queries skipped as before; all evaluation queries retained.',
        arms=dict(original='Read-only adopted22050-parameter3k checkpoints; whole-union selection',
                  capacity='Width64 instead of32, same original32+294 features and loss',
                  visual='Width64 plus160-channel per-view local-content branch from existing raw caches',
                  visual_shuffled='Exact visual architecture/init/batches; fixed within-query within-view permutation of valid candidate local features during both training and evaluation',
                  listwise_only='Exact capacity architecture/init/batches; remove only auxiliary.25 balancedBCE, preserve union multipositive/null listwise loss'),
        parameter_counts={a:sum(p.numel() for p in BottleneckModel(64,a.startswith('visual')).parameters()) for a in ARMS},
        local_feature='Fixed32-D Rademacher projections of normalized1152-D fine vision cell and3x3 neighborhood,4096-D coarse decoder image cell, mean target query at layer15, and projected image/query product.160 channels/view; public point/view geometry only. No newly encoded high-resolution crop.',
        projection_seed=20261901,shuffle='SHA(sample_id|v19_visual_shuffle|view), valid candidate permutation independently in each view; masks and each-view feature multisets preserved.',
        normalization='Original32 base channels mean/std and local160 mean/std computed from fitting-train candidates only; missing local views zeroed after scaling.',
        optimizer=dict(lr=.0002,weight_decay=.02,batch_queries=8,gradient_clip=1.,steps=STEPS,dropout=0),
        primary=dict(cohort='new_development300, historically used development',contrasts=CONTRASTS,
                     intervals='5000 source-stratified RGB-cluster bootstrap draws over fixed3-fit means; Bonferroni4 comparisons,98.75% intervals; no training-seed population interval',practical_threshold_pp=2.),
        secondary='Old172 and train2973 descriptive95% contrasts; per-source, candidate-conditioned accuracy, common train loss, and new-development source-group95% sensitivity.',
        interpretation='Capacity evidence needs capacity>original; aligned-content evidence needs visual>capacity AND visual>visual_shuffled. Loss evidence needs listwise_only>capacity. Require at least+2pp and positive adjusted lower bound on each relevant primary contrast. Null results do not rule out other architectures, optimization or local visual representations.',
        no_selection='All12 new checkpoints frozen before any new evaluation. No best seed/checkpoint selection. No automatic adoption.',
        independent_test='Do not access V18 fresh280 features, labels or predictions. This branch uses only development to diagnose mechanisms; no new MVP/independent superiority claim.',
        execution='Two independent training workers on cuda0/cuda1. Immutable half state/ROI caching on GPU when estimated storage leaves2.5GiB margin, otherwise identical CPU streaming; same numeric inputs and float32 model operations. Resume every250 steps.',
        mainline_unchanged=True,html=False)
    for p in SOURCES:
        dest=OUT/'source_archive'/p.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest)
    write(OUT/'protocol.json',spec);status('registered',training_queries=2973,new_fits=12)
    return spec


def feature_one(task):
    e, public=task;dest=OUT/f'features/cases/{key(e["sample_id"])}.json'
    if dest.exists():
        cached=read(dest)
        for p,h in [('roi_path','roi_sha256'),('roi_mask_path','roi_mask_sha256')]:assert sha(cached[p])==cached[h]
        assert cached['base_sha256']==e['sha256'];return cached
    assert sha(e['path'])==e['sha256'] and sha(e['pipeline_path'])==e['pipeline_sha256']
    pipe=read(e['pipeline_path']);assert not pipe['ground_truth_in_model_inputs']
    with np.load(e['path'],allow_pickle=False) as z:points=z['points'].copy()
    assert len(points)==e['n']
    output=np.zeros((len(points),4,ROI_DIM),np.float16);mask=np.zeros((len(points),4),bool);raw_hashes={}
    views=[dict(feature_path=pipe['global_feature_path'],feature_sha256=pipe['global_feature_sha256'],transform=None)]+pipe['local_views']
    assert len(views)<=4
    if len(points):
        for j,v in enumerate(views):
            assert sha(v['feature_path'])==v['feature_sha256'];raw_hashes[v['feature_path']]=v['feature_sha256']
            with np.load(v['feature_path'],allow_pickle=False) as z:
                meta=json.loads(str(z['metadata']));arrays={k:z[k] for k in ['image_hidden','vision_hidden','target_hidden']}
            authenticate_meta(meta,public,legacy_global=j==0 and 'runtime_identity_sha256' in pipe)
            output[:,j],mask[:,j]=local_view_features(arrays,meta,points,public['image_size'],v.get('transform'))
    paths=[OUT/f'features/arrays/{key(e["sample_id"])}{suffix}' for suffix in ['.roi.npy','.mask.npy']]
    for p,a in zip(paths,[output,mask]):
        p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp')
        with tmp.open('wb') as stream:np.save(stream,a,allow_pickle=False)
        replace(tmp,p)
    entry=dict(sample_id=e['sample_id'],n=e['n'],base_sha256=e['sha256'],roi_path=str(paths[0]),roi_sha256=sha(paths[0]),
               roi_mask_path=str(paths[1]),roi_mask_sha256=sha(paths[1]),raw_sha256=raw_hashes,
               protocol_sha256=sha(OUT/'protocol.json'))
    write(dest,entry);return entry


def extract():
    verify();target=OUT/'features/index.json'
    if target.exists():assert read(target)['complete'];return
    indices=read(OUT/'data/indices.json');public=read(OUT/'data/public.json')
    entries=[e for cohort in indices.values() for e in cohort];started=time.perf_counter();result=[]
    status('extracting_cached_local_features',completed=0,total=len(entries))
    with ThreadPoolExecutor(max_workers=2) as pool:
        for i,e in enumerate(pool.map(feature_one,[(e,public[e['sample_id']]) for e in entries]),1):
            result.append(e)
            if i%20==0 or i==len(entries):status('extracting_cached_local_features',completed=i,total=len(entries),elapsed_seconds=time.perf_counter()-started)
    write(target,dict(complete=True,cases=result,protocol_sha256=sha(OUT/'protocol.json')))
    normalization()


def normalization():
    path=OUT/'normalization.pt'
    if path.exists():assert sha(path)==read(OUT/'normalization.json')['sha256'];return
    entries=read(OUT/'data/indices.json')['train'];roi={e['sample_id']:e for e in read(OUT/'features/index.json')['cases']}
    bases=[];total=np.zeros(ROI_DIM,np.float64);square=total.copy();count=0
    for e in entries:
        if not e['n']:continue
        with np.load(e['path'],allow_pickle=False) as z:bases.append(z['base'].copy())
        r=roi[e['sample_id']];v=np.load(r['roi_path'],allow_pickle=False);mask=np.load(r['roi_mask_path'],allow_pickle=False)
        valid=v[mask].astype(np.float64);total+=valid.sum(0);square+=(valid*valid).sum(0);count+=len(valid)
    base=np.concatenate(bases);mean=base.mean(0);std=base.std(0);std[std<1e-5]=1
    rm=(total/count).astype(np.float32);rs=np.sqrt(np.maximum(square/count-(total/count)**2,0)).astype(np.float32);rs[rs<1e-5]=1
    for s in [1,2,3]:
        ckpt=torch.load(MAIN/f'models/expanded_s{s}.pt',weights_only=True,map_location='cpu')
        assert np.array_equal(mean,ckpt['mean'].numpy()) and np.array_equal(std,ckpt['std'].numpy())
    save(path,dict(mean=torch.from_numpy(mean),std=torch.from_numpy(std),roi_mean=torch.from_numpy(rm),roi_std=torch.from_numpy(rs)))
    write(OUT/'normalization.json',dict(complete=True,path=str(path),sha256=sha(path),training_valid_candidate_views=count,
                                      original_base_scaler_exact_match=True,protocol_sha256=sha(OUT/'protocol.json')))


def load_records(cohort):
    entries=read(OUT/'data/indices.json')[cohort]
    features={e['sample_id']:e for e in read(OUT/'features/index.json')['cases']}
    records=legacy.load_records(entries)
    for r in records:
        e=features[r['sample_id']]
        for p,h in [('roi_path','roi_sha256'),('roi_mask_path','roi_mask_sha256')]:assert sha(e[p])==e[h]
        r['roi']=np.load(e['roi_path'],allow_pickle=False);r['roi_mask']=np.load(e['roi_mask_path'],allow_pickle=False)
        assert r['roi'].shape==(r['n'],4,ROI_DIM) and r['roi_mask'].shape==(r['n'],4)
    return records


class BatchCache:
    def __init__(self,records,norm,device):
        self.device=device;self.records=records;self.data=[]
        need=sum(r['state'].nbytes+r['mask'].nbytes+r['base'].nbytes+r['roi'].nbytes+r['roi_mask'].nbytes+r['n']*4*8 for r in records)
        free=torch.cuda.mem_get_info(device)[0]
        self.storage=device if need<min(12.5*2**30,free-2.5*2**30) else 'cpu'
        self.cache_bytes=need
        mean,std=norm['mean'].numpy(),norm['std'].numpy();rm,rs=norm['roi_mean'].numpy(),norm['roi_std'].numpy()
        for r in records:
            roi=((r['roi'].astype(np.float32)-rm)/rs);roi[~r['roi_mask']]=0
            row=dict(base=torch.tensor((r['base']-mean)/std,device=self.storage),
                     state=torch.tensor(np.array(r['state']),device=self.storage),
                     mask=torch.tensor(np.array(r['mask']),device=self.storage),
                     roi=torch.tensor(roi.astype(np.float16),device=self.storage),
                     roi_mask=torch.tensor(r['roi_mask'],device=self.storage),
                     permutation=torch.tensor(shuffle_indices(r['roi_mask'],r['sample_id']),device=self.storage),
                     labels=torch.tensor(r['labels'],device=device))
            self.data.append(row)

    def batch(self,indices,shuffle=False):
        data=[self.data[int(i)] for i in indices];lengths=[len(d['base']) for d in data]
        tmax=max(d['state'].shape[2] for d in data)
        args=dict(base=torch.cat([d['base'] for d in data]),
                  state=torch.cat([F.pad(d['state'],(0,0,0,tmax-d['state'].shape[2])) for d in data]),
                  mask=torch.cat([F.pad(d['mask'],(0,tmax-d['mask'].shape[2])) for d in data]),
                  roi=torch.cat([d['roi'][d['permutation'],torch.arange(4,device=self.storage)[None,:]] if shuffle else d['roi'] for d in data]),
                  roi_mask=torch.cat([d['roi_mask'] for d in data]))
        return {k:v.to(self.device) for k,v in args.items()},lengths,[d['labels'] for d in data]


def model_for(arm):
    return RetentionModel() if arm=='original' else BottleneckModel(64,arm in ['visual','visual_shuffled'])


def forward(model,args):
    return model(**({k:v for k,v in args.items() if k in ['base','state','mask']} if isinstance(model,RetentionModel) else args))


def state_identity(model):
    return digest({k:hashlib.sha256(v.detach().cpu().numpy().tobytes()).hexdigest() for k,v in model.state_dict().items()})


def train_fit(arm,seed,cache,worker):
    name=f'{arm}_s{SEEDS.index(seed)+1}';receipt=OUT/f'models/{name}.json'
    if receipt.exists():assert sha(read(receipt)['checkpoint_path'])==read(receipt)['checkpoint_sha256'];return
    torch.manual_seed(seed);np.random.seed(seed);model=model_for(arm).to(cache.device);initial=state_identity(model)
    optimizer=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=.02)
    identity=dict(protocol_sha256=sha(OUT/'protocol.json'),features_sha256=sha(OUT/'features/index.json'),
                  normalization_sha256=sha(OUT/'normalization.pt'),arm=arm,seed=seed)
    resume=OUT/f'models/{name}.resume.pt';step=0;epoch=1;position=0;elapsed=0.;history=[]
    if resume.exists():
        saved=torch.load(resume,weights_only=True,map_location='cpu');assert saved['identity']==identity
        model.load_state_dict(saved['state_dict']);optimizer.load_state_dict(saved['optimizer'])
        step,epoch,position,elapsed,history=[saved[k] for k in ['step','epoch','position','elapsed','history']]
        torch.set_rng_state(saved['torch_rng']);torch.cuda.set_rng_state(saved['cuda_rng'],cache.device)
    fitting=[i for i,r in enumerate(cache.records) if r['n']];started=time.perf_counter();losses=[];model.train()
    status('training',worker=worker,model=name,step=step,total_steps=STEPS,gpu_cache=cache.storage,cache_GiB=cache.cache_bytes/2**30)
    while step<STEPS:
        ix,epoch,position=legacy.next_batch(len(fitting),seed,epoch,position);selected=[fitting[int(i)] for i in ix]
        args,lengths,labels=cache.batch(selected,arm=='visual_shuffled');optimizer.zero_grad(set_to_none=True)
        scores=forward(model,args)
        loss=torch.stack([probe_loss(s,y,model.null,0 if arm=='listwise_only' else .25) for s,y in zip(scores.split(lengths),labels)]).mean()
        assert torch.isfinite(loss),'nonfinite loss';loss.backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()),'nonfinite gradient'
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();step+=1;losses.append(float(loss.detach()))
        if step%100==0 or step==STEPS:
            h=dict(step=step,epoch=epoch,position=position,train_loss=float(np.mean(losses)),elapsed_seconds=elapsed+time.perf_counter()-started)
            history.append(h);losses=[];status('training',worker=worker,model=name,total_steps=STEPS,**h)
        if step%250==0 or step==STEPS:
            save(resume,dict(identity=identity,state_dict=model.state_dict(),optimizer=optimizer.state_dict(),step=step,epoch=epoch,
                             position=position,elapsed=elapsed+time.perf_counter()-started,history=history,
                             torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(cache.device)))
    checkpoint=OUT/f'models/{name}.pt'
    save(checkpoint,dict(state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},arm=arm,seed=seed,identity=identity))
    write(receipt,dict(complete=True,name=name,arm=arm,seed=seed,steps=STEPS,parameter_count=sum(p.numel() for p in model.parameters()),
                       fitting_queries=len(fitting),manifest_queries=len(cache.records),empty_skipped=len(cache.records)-len(fitting),
                       initial_state_sha256=initial,checkpoint_path=str(checkpoint),checkpoint_sha256=sha(checkpoint),
                       identity=identity,history=history,validation_used_for_selection=False))
    del model,optimizer;gc.collect();torch.cuda.empty_cache()


def worker_run(worker):
    verify();device=f'cuda:{worker}';torch.cuda.set_device(device);torch.set_num_threads(2)
    status('loading_train',worker=worker);records=load_records('train');norm=torch.load(OUT/'normalization.pt',weights_only=True)
    cache=BatchCache(records,norm,device)
    for arm in (['capacity','listwise_only'] if worker==0 else ['visual','visual_shuffled']):
        for seed in SEEDS:train_fit(arm,seed,cache,worker)
    status('complete',worker=worker)


def train_all():
    verify();normalization();workers=[]
    for worker in [0,1]:
        p=OUT/f'workers/{worker}';p.parent.mkdir(parents=True,exist_ok=True)
        stdout=Path(str(p)+'.stdout.log').open('a',encoding='utf-8');stderr=Path(str(p)+'.stderr.log').open('a',encoding='utf-8')
        proc=subprocess.Popen([sys.executable,'-B','-X','utf8','-u',str(Path(__file__).resolve()),'--stage','worker','--worker',str(worker)],cwd=ROOT,stdout=stdout,stderr=stderr)
        stdout.close();stderr.close();workers.append(proc)
    write(OUT/'workers/execution.json',dict(started_at_utc=now(),pids=[p.pid for p in workers]))
    while True:
        states=[p.poll() for p in workers]
        progress={str(i):read(OUT/f'workers/{i}.status.json') if (OUT/f'workers/{i}.status.json').exists() else {} for i in [0,1]}
        status('training_workers',exit_codes=states,workers=progress)
        if any(code is not None and code!=0 for code in states):
            # Do not create duplicate GPU workers. The surviving worker can finish.
            raise RuntimeError('Training worker failed; inspect its preserved stderr and active sibling before resuming')
        if all(code==0 for code in states):break
        time.sleep(50)
    registry={}
    adopted=read(MAIN/'models/frozen.json')['models']
    for s in [1,2,3]:
        registry[f'original_s{s}']=dict(adopted[f'expanded_s{s}'],arm='original')
        for arm in ARMS:
            row=read(OUT/f'models/{arm}_s{s}.json');assert row['complete'] and row['steps']==STEPS
            assert sha(row['checkpoint_path'])==row['checkpoint_sha256']
            registry[f'{arm}_s{s}']=dict(path=row['checkpoint_path'],sha256=row['checkpoint_sha256'],arm=arm)
        for a,b in [('capacity','listwise_only'),('visual','visual_shuffled')]:
            assert read(OUT/f'models/{a}_s{s}.json')['initial_state_sha256']==read(OUT/f'models/{b}_s{s}.json')['initial_state_sha256']
    for row in registry.values():assert sha(row['path'])==row['sha256']
    write(OUT/'models/frozen.json',dict(complete=True,frozen_at_utc=now(),models=registry,protocol_sha256=sha(OUT/'protocol.json')))
    status('models_frozen',new_fits=12,model_slots=15)


@torch.inference_mode()
def evaluate():
    verify();frozen=read(OUT/'models/frozen.json');assert frozen['complete']
    device='cuda:0';torch.cuda.set_device(device);torch.set_num_threads(2)
    norm=torch.load(OUT/'normalization.pt',weights_only=True);models={}
    for name,row in frozen['models'].items():
        assert sha(row['path'])==row['sha256'];saved=torch.load(row['path'],weights_only=True,map_location='cpu')
        model=model_for(row['arm']).to(device).eval();model.load_state_dict(saved['state_dict']);models[name]=model
    for cohort in ['train','old_val','new_development']:
        path=OUT/f'evaluation/{cohort}.json'
        if path.exists():assert read(path)['complete'];continue
        status('loading_evaluation',cohort=cohort);records=load_records(cohort);cache=BatchCache(records,norm,device)
        vectors={n:[] for n in models};losses={n:[] for n in models};conditional={n:[] for n in models}
        replay_agreement=replay_total=0;max_score_error=0.
        for start in range(0,len(records),8):
            indices=list(range(start,min(start+8,len(records))));args,lengths,labels=cache.batch(indices)
            shuffled_args=None
            for name,model in models.items():
                if name.startswith('visual_shuffled'):
                    if shuffled_args is None:shuffled_args=cache.batch(indices,True)[0]
                    inputs=shuffled_args
                else:inputs=args
                score_chunks=forward(model,inputs).split(lengths)
                for i,s,y in zip(indices,score_chunks,labels):
                    r=records[i];pred=int(s.argmax()) if len(s) else None;hit=bool(pred is not None and y[pred])
                    vectors[name].append(hit)
                    if bool(y.any()):conditional[name].append(hit)
                    if len(s):losses[name].append(float(probe_loss(s,y,model.null,.25)))
                    if cohort!='train' and name.startswith('original'):
                        prior='old_val' if cohort=='old_val' else 'new_reserve'
                        p=MAIN/f'predictions/{prior}/expanded_s{name[-1]}/{key(r["sample_id"])}.npz'
                        with np.load(p,allow_pickle=False) as z:reference=z['scores'].copy()
                        scores=s.float().cpu().numpy()
                        assert reference.shape==scores.shape
                        error=float(np.max(np.abs(reference-scores))) if len(scores) else 0.
                        max_score_error=max(max_score_error,error)
                        replay_total+=1;replay_agreement+=int((int(reference.argmax()) if len(reference) else None)==pred)
            if start%80==0 or start+8>=len(records):status('evaluating',cohort=cohort,completed=min(start+8,len(records)),total=len(records))
        oracle=sum(bool(np.any(r['labels'])) for r in records)
        methods={name:dict(hits=sum(v),accuracy_percent=100*float(np.mean(v)),
                           oracle_hits=oracle,conditional_accuracy_percent=100*float(np.mean(conditional[name])),
                           common_loss=float(np.mean(losses[name]))) for name,v in vectors.items()}
        assert replay_agreement==replay_total and max_score_error<1e-4,'Original replay mismatch'
        write(path,dict(complete=True,cohort=cohort,n=len(records),methods=methods,hit_vectors=vectors,
                        sample_ids=[r['sample_id'] for r in records],oracle_hits=oracle,
                        original_replay=dict(point_agreement=replay_agreement,total=replay_total,max_score_error=max_score_error),
                        model_freeze_sha256=sha(OUT/'models/frozen.json')))
        del cache,records,args,shuffled_args;gc.collect();torch.cuda.empty_cache()


def bootstrap(values,entries,level,group_field='group'):
    strata={}
    for i,e in enumerate(entries):strata.setdefault(e['source'],{}).setdefault(e[group_field],[]).append(i)
    clusters=[list(d.values()) for d in strata.values()];rng=np.random.default_rng(20261902)
    draws=[[i for groups in clusters for j in rng.integers(0,len(groups),len(groups)) for i in groups[j]] for _ in range(5000)]
    result={}
    for name,(a,b) in CONTRASTS.items():
        diff=(values[a]-values[b]).mean(0);alpha=1-level
        ci=100*np.quantile([diff[ix].mean() for ix in draws],[alpha/2,1-alpha/2]);delta=100*float(diff.mean())
        result[name]=dict(arms=[a,b],delta_pp=delta,ci_pp=ci.tolist(),interval_percent=100*level,
                          seed_deltas_pp=(100*(values[a]-values[b]).mean(1)).tolist(),
                          meets_registered_positive_gate=bool(delta>=2 and ci[0]>0))
    return result


def analyze():
    verify();indices=read(OUT/'data/indices.json');results={};files=[]
    lines=['V19 固定候选/数据的瓶颈区分：容量、局部视觉对应、辅助损失。',
           '开发诊断，不是新独立测试；不访问刚完成的280张测试，不挑选种子或自动替换主线。',
           '训练2973条清单，固定2976步/批8/同优化器，12个新拟合，三个种子均报告。',
           '局部特征来自原缓存的细网格视觉向量与图文向量；没有新增高分辨率图像编码。']
    for cohort in ['train','old_val','new_development']:
        path=OUT/f'evaluation/{cohort}.json';r=read(path);assert r['complete'];files.append(path)
        entries=indices[cohort];values={arm:np.asarray([r['hit_vectors'][f'{arm}_s{s}'] for s in [1,2,3]],float) for arm in ['original']+ARMS}
        methods={arm:dict(accuracy_percent=100*float(v.mean()),seed_hits=v.sum(1).astype(int).tolist(),
                          conditional_accuracy_percent=100*float(v.sum(1).mean())/r['oracle_hits'],
                          common_loss=float(np.mean([r['methods'][f'{arm}_s{s}']['common_loss'] for s in [1,2,3]])),
                          by_source={source:100*float(v[:,[i for i,e in enumerate(entries) if e['source']==source]].mean()) for source in sorted({e['source'] for e in entries})}) for arm,v in values.items()}
        comparisons=bootstrap(values,entries,.9875 if cohort=='new_development' else .95)
        results[cohort]=dict(n=len(entries),oracle_hits=r['oracle_hits'],methods=methods,comparisons=comparisons)
        if cohort=='new_development':results[cohort]['source_group_sensitivity95']=bootstrap(values,entries,.95,'source_group')
        lines += ['',f'{cohort}: n={len(entries)}, 固定候选可达{r["oracle_hits"]}']
        for arm,row in methods.items():lines.append(f'{arm}: {row["accuracy_percent"]:.2f}%, 三种子{row["seed_hits"]}; 可达条件命中{row["conditional_accuracy_percent"]:.2f}%; 共同损失{row["common_loss"]:.4f}')
        for name,row in comparisons.items():lines.append(f'{name}: {row["delta_pp"]:+.2f}pp, {row["interval_percent"]:.2f}%区间{row["ci_pp"]}; 达到登记门槛={row["meets_registered_positive_gate"]}')
    comp=results['new_development']['comparisons']
    interpretation=dict(capacity_supported=comp['capacity']['meets_registered_positive_gate'],
                        aligned_visual_probe_supported=comp['visual_vs_capacity']['meets_registered_positive_gate'] and comp['visual_alignment']['meets_registered_positive_gate'],
                        listwise_only_supported=comp['loss_objective']['meets_registered_positive_gate'])
    lines += ['', '登记条件下的判断：'+json.dumps(interpretation,ensure_ascii=False),
              '不显著不等于证明该机制不存在；固定步数/LR、投影维数、原缓存分辨率和开发来源限制了解释。',
              'visual与capacity的参数量不同，因此必须同时看visual对同结构同初始化visual_shuffled的对照；只看前者不能归因于视觉信息。',
              '训练更好但开发无提升提示泛化/分布或优化限制，不能直接断言唯一成因。',
              '原3k三个权重未修改；没有VLM新推理，没有HTML看板，没有独立超越MVP的主张。']
    write(OUT/'results.json',dict(complete=True,cohorts=results,interpretation=interpretation,mainline_unchanged=True,independent_confirmation=False))
    (OUT/'results.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    adopted=read(MAIN/'models/frozen.json')['models']
    for s in [1,2,3]:assert sha(adopted[f'expanded_s{s}']['path'])==adopted[f'expanded_s{s}']['sha256']
    verify();files += [OUT/'protocol.json',OUT/'features/index.json',OUT/'normalization.json',OUT/'models/frozen.json',OUT/'results.json',OUT/'results.txt']
    write(OUT/'completion.json',dict(complete=True,completed_at_utc=now(),artifact_sha256={str(p):sha(p) for p in files},mainline_unchanged=True))
    status('complete',interpretation=interpretation);print('\n'.join(lines),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['prepare','features','worker','train','evaluate','analyze','all'],default='all');p.add_argument('--worker',type=int,choices=[0,1]);a=p.parse_args()
    torch.set_num_threads(2)
    try:
        prepare()
        if a.stage in ['features','all']:extract()
        if a.stage=='worker':worker_run(a.worker)
        if a.stage in ['train','all']:train_all()
        if a.stage in ['evaluate','all']:evaluate()
        if a.stage in ['analyze','all']:analyze()
    except BaseException:
        status('failed',worker=a.worker if a.stage=='worker' else None,error=traceback.format_exc());raise


if __name__=='__main__':main()
