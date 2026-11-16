"""Registered local V20 experiment. No HTML, backbone calls or held-out test."""
import os
os.environ.setdefault('PYTHONDONTWRITEBYTECODE', '1')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
import argparse
import gc
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
import run_bottleneck_v19 as prior
from clickagent.transfer_alignment_v20 import TransferMatcher, pair_loss
from clickagent.bottleneck_v19 import probe_loss, shuffle_indices
from clickagent.evidence_retention import RetentionModel

OUT = ROOT/'runs/transfer_alignment_v20'
SEEDS = prior.SEEDS
STEPS = 2976
ARMS = ['matching', 'paired', 'regularized', 'paired_regularized']
COHORTS = ['train', 'old_val', 'new_development']
CONTRASTS = {'combined_vs_original': ('paired_regularized', 'original'),
             'combined_vs_matching': ('paired_regularized', 'matching'),
             'pairing_effect': ('paired', 'matching'),
             'regularization_effect': ('regularized', 'matching')}
read, write, sha, now, save = prior.read, prior.write, prior.sha, prior.now, prior.save
SOURCES = [Path(__file__), ROOT/'src/clickagent/transfer_alignment_v20.py', ROOT/'tests/test_transfer_alignment_v20.py']


def status(phase, worker=None, **fields):
    row = dict(phase=phase, complete=phase=='complete', updated_at_utc=now(), **fields)
    write(OUT/('status.json' if worker is None else f'workers/{worker}.status.json'), row)
    print(json.dumps(row, ensure_ascii=False), flush=True)


def verify():
    spec = read(OUT/'protocol.json')
    for p,h in spec['hashes'].items():
        assert sha(p)==h, p
    return spec


def prepare():
    if (OUT/'protocol.json').exists():
        return verify()
    prior.verify()
    inputs = SOURCES + [ROOT/'scripts/run_bottleneck_v19.py', ROOT/'src/clickagent/bottleneck_v19.py',
              ROOT/'src/clickagent/evidence_retention.py', ROOT/'scripts/run_quality_readout_v15.py']
    inputs += [prior.OUT/p for p in ['protocol.json','data/indices.json','data/public.json','features/index.json',
                                      'normalization.pt','models/frozen.json','results.json']]
    inputs += [prior.OUT/f'evaluation/{c}.json' for c in COHORTS]
    adopted = read(prior.MAIN/'models/frozen.json')['models']
    inputs += [Path(adopted[f'expanded_s{s}']['path']) for s in [1,2,3]]
    spec = dict(registered_at_utc=now(), hashes={str(p):sha(p) for p in inputs},
        fixed='Original2973 manifest/2878 nonempty fits, same fine candidate pools, frozen backbone/views/coordinates and original3k heads.2976 updates,batch8,3 seeds,final step only.',
        seeds=list(SEEDS), steps=STEPS, arms=ARMS,
        model='4097-parameter shared linear region96->32/query32->32 cosine matching residual. Region fine32+neighborhood32+decoder32, query layer15 projected32. No IDs, coordinates, source or labels in model. Residual4*tanh(alpha)*mean_valid_view_cosine, alpha starts0. Original22050-param head including null frozen; total26147, trainable4097.',
        initialization='Both maps near shared semantic identity, independent Gaussian.001 perturbation; same initial weights and batch order for all four arms at each seed.',
        pairing='paired arms add.1*symmetric InfoNCE(temp.1) on batch native global queries and mean GT-positive region prototypes. Exclude no-positive queries, sameRGB negatives and exact normalized instruction negatives. Other semantic false negatives remain possible. Native cached region representation is instruction-conditioned; no full counterfactual re-encoding claim.',
        regularization='regularized arms use.2 raw feature channel dropout, independent region/query masks perquery/view shared across candidates; plus.1*sum squared deviation from initial maps/32. This is a regularization bundle, not separate dropout/anchor attribution.',
        optimizer=dict(lr=.0002,weight_decay=.02,batch_queries=8,gradient_clip=1.,steps=STEPS),
        objective='Original multipositive/null listwise+.25 balancedBCE on frozen original logits+residual; pairing/regularization only as above. No early stopping, no hyperparameter sweep.',
        normalization='Use raw128 half-precision V19 projected channels, L2-normalized learned32-D embeddings. Do not feed V19 product32 or standardized query-only MLP features.',
        primary=dict(cohort='Historically reused new_development300, not independent',contrasts=CONTRASTS,
                     bootstrap='5000 source-stratified RGB-cluster bootstrap; fixed3-fit mean differences,98.75% Bonferroni4 intervals',positive_gate_pp=2.),
        guard='Old172 reported with95% cluster intervals; candidate advancement also requires old point effect>=-2pp. No automatic adoption.',
        diagnostics='Train accuracy/conditional accuracy and common loss; train-development gaps; factorial interaction; per-source and source-group cluster sensitivity. Fixed-weight global-view wrong-query margin probe (same source,differentRGB and nonidentical instruction) and spatial permutation residual probe. Partial feature interventions, not end-to-end instruction swaps.',
        independent_test='V18 fresh280 completely excluded. Positive development outcomes require a new independent test later; no MVP superiority claim.',
        freeze='All12 new checkpoints frozen before any new development evaluation; report every seed mean, no ensemble.',
        execution='Compute frozen original scores once with exact V19 hit replay, then train compact local features on2 GPUs. No new backbone inference. Python logs/JSON/TXT/PNG,no HTML.')
    for p in SOURCES:
        dest=OUT/'source_archive'/p.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest)
    write(OUT/'protocol.json',spec)
    write(OUT/'registry_before.json',read(ROOT/'CURRENT_RESEARCH.json'))
    registry=read(ROOT/'CURRENT_RESEARCH.json')
    registry['active_experiment']=dict(run_id=OUT.name,status='registered',protocol_path=str(OUT/'protocol.json'),
         status_path=str(OUT/'status.json'),new_fits=12,mainline_unchanged=True,output_format='Python/JSON/TXT/PNG,no HTML')
    write(ROOT/'CURRENT_RESEARCH.json',registry)
    write(ROOT/'research/decisions/20261108_transfer_alignment_v20.json',dict(user_request='试试学习可迁移的指令—区域对应，抑制样本记忆这个方向吧',
         recorded_at_utc=now(),protocol_path=str(OUT/'protocol.json'),protocol_sha256=sha(OUT/'protocol.json'),mainline_unchanged=True))
    status('registered',new_fits=12)
    return spec


@torch.inference_mode()
def compact():
    verify();torch.cuda.set_device(0);torch.set_num_threads(2)
    public=read(prior.OUT/'data/public.json');norm=torch.load(prior.OUT/'normalization.pt',weights_only=True)
    adopted=read(prior.MAIN/'models/frozen.json')['models'];models=[];nulls=[]
    for s in [1,2,3]:
        row=adopted[f'expanded_s{s}'];assert sha(row['path'])==row['sha256']
        model=RetentionModel().cuda().eval();model.load_state_dict(torch.load(row['path'],weights_only=True,map_location='cpu')['state_dict'])
        models.append(model);nulls.append(float(model.null))
    receipts={}
    for cohort in COHORTS:
        dest=OUT/f'cache/{cohort}.pt';receipt=dest.with_suffix('.json')
        if receipt.exists():
            r=read(receipt);assert sha(dest)==r['sha256'];receipts[cohort]=r;continue
        status('loading_frozen_features',cohort=cohort)
        records=prior.load_records(cohort);cache=prior.BatchCache(records,norm,'cuda:0');rows=[];replay=0
        reference=read(prior.OUT/f'evaluation/{cohort}.json')
        for start in range(0,len(records),8):
            ix=list(range(start,min(start+8,len(records))));args,lengths,labels=cache.batch(ix)
            chunks=[model(**{k:args[k] for k in ['base','state','mask']}).split(lengths) for model in models]
            for j,i in enumerate(ix):
                r=records[i];scores=torch.stack([chunk[j].cpu().clone() for chunk in chunks])
                for s in range(3):
                    hit=bool(len(r['labels']) and r['labels'][int(scores[s].argmax())])
                    assert hit==reference['hit_vectors'][f'original_s{s+1}'][i];replay+=1
                rows.append(dict(sample_id=r['sample_id'],source=r['source'],group=r['group'],source_group=r['source_group'],
                    instruction=' '.join(public[r['sample_id']]['instruction'].casefold().split()),n=r['n'],
                    roi=torch.from_numpy(r['roi'][...,:128].copy()),roi_mask=torch.from_numpy(r['roi_mask'].copy()),
                    labels=torch.from_numpy(r['labels'].copy()).bool(),scores=scores))
            if start%160==0:status('caching_frozen_scores',cohort=cohort,completed=min(start+8,len(records)),total=len(records))
        save(dest,dict(rows=rows,nulls=nulls,protocol_sha256=sha(OUT/'protocol.json')))
        r=dict(complete=True,path=str(dest),sha256=sha(dest),n=len(rows),original_hit_replay=replay,
               oracle=sum(bool(r['labels'].any()) for r in rows),protocol_sha256=sha(OUT/'protocol.json'))
        write(receipt,r);receipts[cohort]=r
        del cache,records,args,rows,chunks;gc.collect();torch.cuda.empty_cache()
    write(OUT/'cache/frozen.json',dict(complete=True,cohorts=receipts,protocol_sha256=sha(OUT/'protocol.json')))
    status('compact_features_frozen',original_hit_replay=sum(r['original_hit_replay'] for r in receipts.values()))


def load_compact(cohort,device):
    receipt=read(OUT/f'cache/{cohort}.json');assert sha(receipt['path'])==receipt['sha256']
    saved=torch.load(receipt['path'],weights_only=True,map_location=device)
    assert saved['protocol_sha256']==sha(OUT/'protocol.json')
    return saved


def batch(rows,ix,seed_index):
    selected=[rows[int(i)] for i in ix];lengths=[r['n'] for r in selected]
    roi=torch.cat([r['roi'] for r in selected]);device=roi.device
    ids=torch.repeat_interleave(torch.arange(len(ix),device=device),torch.tensor(lengths,device=device))
    return dict(roi=roi,roi_mask=torch.cat([r['roi_mask'] for r in selected]),query_ids=ids,queries=len(ix)),lengths,selected,torch.cat([r['scores'][seed_index] for r in selected])


def train_fit(arm,seed,saved,worker):
    s=SEEDS.index(seed);name=f'{arm}_s{s+1}';receipt=OUT/f'models/{name}.json'
    if receipt.exists():assert sha(read(receipt)['path'])==read(receipt)['sha256'];return
    device=f'cuda:{worker}';torch.manual_seed(seed);np.random.seed(seed)
    model=TransferMatcher('regularized' in arm).to(device);initial=prior.state_identity(model)
    optimizer=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=.02)
    identity=dict(protocol=sha(OUT/'protocol.json'),cache=sha(OUT/'cache/frozen.json'),arm=arm,seed=seed)
    rows=saved['rows'];null=torch.tensor(saved['nulls'][s],device=device);fitting=[i for i,r in enumerate(rows) if r['n']]
    step=0;epoch=1;position=0;elapsed=0.;history=[];resume=OUT/f'models/{name}.resume.pt'
    if resume.exists():
        old=torch.load(resume,weights_only=True,map_location='cpu');assert old['identity']==identity
        model.load_state_dict(old['state']);optimizer.load_state_dict(old['optimizer'])
        step,epoch,position,elapsed,history=[old[k] for k in ['step','epoch','position','elapsed','history']]
        torch.set_rng_state(old['rng']);torch.cuda.set_rng_state(old['cuda_rng'],device)
    model.train();started=time.perf_counter();losses=[]
    status('training',worker=worker,model=name,step=step,total_steps=STEPS)
    while step<STEPS:
        ix,epoch,position=prior.legacy.next_batch(len(fitting),seed,epoch,position)
        args,lengths,selected,baseline=batch(rows,[fitting[int(i)] for i in ix],s)
        optimizer.zero_grad(set_to_none=True);residual,rr,qq=model(**args)
        labels=[r['labels'] for r in selected]
        rank=torch.stack([probe_loss(z,y,null,.25) for z,y in zip((baseline+residual).split(lengths),labels)]).mean()
        pair,eligible=pair_loss(rr,qq,lengths,labels,[r['group'] for r in selected],[r['instruction'] for r in selected])
        anchor=model.anchor_penalty()
        loss=rank+(.1*pair if arm in ['paired','paired_regularized'] else 0)+(.1*anchor if 'regularized' in arm else 0)
        assert torch.isfinite(loss);loss.backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();step+=1
        losses.append([float(rank.detach()),float(pair.detach()),float(anchor.detach()),eligible])
        if step%100==0 or step==STEPS:
            means=np.mean(losses,axis=0);losses=[]
            h=dict(step=step,epoch=epoch,position=position,elapsed_seconds=elapsed+time.perf_counter()-started,
                   rank_loss=float(means[0]),pair_loss=float(means[1]),anchor_penalty=float(means[2]),
                   pair_eligible_per_batch=float(means[3]),residual_scale=float(4*model.alpha.detach().tanh()))
            history.append(h);status('training',worker=worker,model=name,total_steps=STEPS,**h)
        if step%250==0 or step==STEPS:
            save(resume,dict(identity=identity,state=model.state_dict(),optimizer=optimizer.state_dict(),step=step,
                 epoch=epoch,position=position,elapsed=elapsed+time.perf_counter()-started,history=history,
                 rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(device)))
    dest=OUT/f'models/{name}.pt';save(dest,dict(state_dict={k:v.cpu() for k,v in model.state_dict().items()},arm=arm,seed=seed,identity=identity))
    write(receipt,dict(complete=True,path=str(dest),sha256=sha(dest),arm=arm,seed=seed,steps=STEPS,
                      fitting_queries=len(fitting),manifest_queries=len(rows),trainable_parameters=sum(p.numel() for p in model.parameters()),
                      initial_state_sha256=initial,history=history,development_used_for_selection=False))
    del model,optimizer;gc.collect();torch.cuda.empty_cache()


def worker_run(worker):
    verify();device=f'cuda:{worker}';torch.cuda.set_device(device);torch.set_num_threads(2)
    status('loading_compact_train',worker=worker);saved=load_compact('train',device)
    for arm in (['matching','paired'] if worker==0 else ['regularized','paired_regularized']):
        for seed in SEEDS:train_fit(arm,seed,saved,worker)
    status('complete',worker=worker)


def train_all():
    verify();workers=[]
    for worker in [0,1]:
        path=OUT/f'workers/{worker}';path.parent.mkdir(parents=True,exist_ok=True)
        with Path(str(path)+'.stdout.log').open('a',encoding='utf-8') as stdout, Path(str(path)+'.stderr.log').open('a',encoding='utf-8') as stderr:
            workers.append(subprocess.Popen([sys.executable,'-B','-X','utf8','-u',str(Path(__file__).resolve()),'--stage','worker','--worker',str(worker)],cwd=ROOT,stdout=stdout,stderr=stderr))
    write(OUT/'workers/execution.json',dict(started_at_utc=now(),pids=[p.pid for p in workers]))
    while True:
        codes=[p.poll() for p in workers]
        progress={str(i):read(OUT/f'workers/{i}.status.json') if (OUT/f'workers/{i}.status.json').exists() else {} for i in [0,1]}
        status('training_workers',exit_codes=codes,workers=progress)
        if any(c is not None and c!=0 for c in codes):raise RuntimeError('Worker failed; inspect stderr and surviving sibling before resuming')
        if all(c==0 for c in codes):break
        time.sleep(50)
    models={}
    for s in [1,2,3]:
        initials=[]
        for arm in ARMS:
            r=read(OUT/f'models/{arm}_s{s}.json');assert r['complete'] and r['steps']==STEPS and sha(r['path'])==r['sha256']
            models[f'{arm}_s{s}']=r;initials.append(r['initial_state_sha256'])
        assert len(set(initials))==1
    write(OUT/'models/frozen.json',dict(complete=True,frozen_at_utc=now(),models=models,protocol_sha256=sha(OUT/'protocol.json')))
    status('models_frozen',new_fits=12)


def wrong_query_indices(rows):
    donors=[]
    for i,r in enumerate(rows):
        options=[j for j,d in enumerate(rows) if d['n'] and d['source']==r['source'] and d['group']!=r['group'] and d['instruction']!=r['instruction']]
        donors.append(options[int(prior.key(r['sample_id']),16)%len(options)] if options else None)
    return donors


@torch.inference_mode()
def evaluate():
    verify();frozen=read(OUT/'models/frozen.json');assert frozen['complete'];device='cuda:0';torch.cuda.set_device(0);torch.set_num_threads(2)
    models={}
    for name,r in frozen['models'].items():
        assert sha(r['path'])==r['sha256'];model=TransferMatcher('regularized' in r['arm']).to(device).eval()
        model.load_state_dict(torch.load(r['path'],weights_only=True,map_location='cpu')['state_dict']);models[name]=model
    for cohort in COHORTS:
        path=OUT/f'evaluation/{cohort}.json'
        if path.exists():assert read(path)['complete'];continue
        status('loading_compact_evaluation',cohort=cohort);saved=load_compact(cohort,device);rows=saved['rows'];donors=wrong_query_indices(rows)
        names=['original_s1','original_s2','original_s3']+list(models)
        hits={n:[] for n in names};losses={n:[] for n in names};shuffled_hits={n:[] for n in models}
        margins={n:[] for n in models};swapped={n:[] for n in models};disagreements={n:0 for n in models}
        for i,r in enumerate(rows):
            n=r['n'];y=r['labels'];roi=r['roi'];mask=r['roi_mask']
            perm=torch.tensor(shuffle_indices(mask.cpu().numpy(),r['sample_id']),device=device)
            for s in [0,1,2]:
                z=r['scores'][s];name=f'original_s{s+1}';hits[name].append(bool(n and y[int(z.argmax())]))
                if n:losses[name].append(float(probe_loss(z,y,torch.tensor(saved['nulls'][s],device=device),.25)))
            for name,model in models.items():
                s=int(name[-1])-1;residual,rr,qq=model(roi,mask);z=r['scores'][s]+residual
                hit=bool(n and y[int(z.argmax())]);hits[name].append(hit)
                if n:
                    losses[name].append(float(probe_loss(z,y,torch.tensor(saved['nulls'][s],device=device),.25)))
                    disagreements[name]+=int(int(z.argmax())!=int(r['scores'][s].argmax()))
                rp=roi[perm,torch.arange(4,device=device)[None,:]]
                rz=model(rp,mask)[0]+r['scores'][s]
                shuffled_hits[name].append(bool(n and y[int(rz.argmax())]))
                if n and bool(y.any()) and bool((~y).any()) and donors[i] is not None:
                    # Global-view-only intervention keeps regions/coordinates/base
                    # fixed. Native image vectors can still encode the instruction.
                    align=(rr[:,0]*qq[0,0]).sum(-1)
                    donor=rows[donors[i]]['roi'][:1]
                    wrong=model.encode(donor)[1][0,0]
                    mismatch=(rr[:,0]*wrong).sum(-1)
                    margins[name].append(float(align[y].mean()-align[~y].mean()))
                    swapped[name].append(float(mismatch[y].mean()-mismatch[~y].mean()))
            if i%160==0 or i+1==len(rows):status('evaluating',cohort=cohort,completed=i+1,total=len(rows))
        old=read(prior.OUT/f'evaluation/{cohort}.json')
        for s in [1,2,3]:assert hits[f'original_s{s}']==old['hit_vectors'][f'original_s{s}']
        write(path,dict(complete=True,cohort=cohort,n=len(rows),sample_ids=[r['sample_id'] for r in rows],
             oracle_hits=sum(bool(r['labels'].any()) for r in rows),hit_vectors=hits,
             common_losses={n:float(np.mean(v)) for n,v in losses.items()},shuffled_hit_vectors=shuffled_hits,
             global_margin_aligned=margins,global_margin_wrong_query=swapped,changed_predictions=disagreements,
             wrong_query_donors=donors,model_freeze_sha256=sha(OUT/'models/frozen.json')))
        del saved,rows,roi,rp,rr,qq;gc.collect();torch.cuda.empty_cache()


def bootstrap(values,entries,level,group_field='group'):
    strata={}
    for i,e in enumerate(entries):strata.setdefault(e['source'],{}).setdefault(e[group_field],[]).append(i)
    clusters=[list(v.values()) for v in strata.values()];rng=np.random.default_rng(20262001)
    diffs={k:(values[a]-values[b]).mean(0) for k,(a,b) in CONTRASTS.items()};samples={k:[] for k in diffs}
    for _ in range(5000):
        ix=[i for groups in clusters for j in rng.integers(0,len(groups),len(groups)) for i in groups[j]]
        for k,d in diffs.items():samples[k].append(d[ix].mean())
    alpha=1-level
    return {k:dict(delta_pp=100*float(d.mean()),ci_pp=(100*np.quantile(samples[k],[alpha/2,1-alpha/2])).tolist(),
                   interval_percent=100*level,seed_deltas_pp=(100*(values[CONTRASTS[k][0]]-values[CONTRASTS[k][1]]).mean(1)).tolist()) for k,d in diffs.items()}


def analyze():
    verify();indices=read(prior.OUT/'data/indices.json');results={}
    lines=['V20 可迁移指令—区域匹配与抑制样本记忆：开发诊断，非独立验证。',
           '固定原2973条/实际2878非空训练，2976步，四组×三种子；原3k评分器冻结。',
           '新增4097参数双线性余弦匹配残差；paired加配对监督，regularized加通道dropout与参数锚定。',
           '所有12个终点模型先冻结，再统一开发评估；不挑种子，不集成，不自动替换主线。']
    for cohort in COHORTS:
        r=read(OUT/f'evaluation/{cohort}.json');entries=indices[cohort]
        values={a:np.array([r['hit_vectors'][f'{a}_s{s}'] for s in [1,2,3]],float) for a in ['original']+ARMS}
        methods={a:dict(accuracy_percent=100*float(v.mean()),seed_hits=v.sum(1).astype(int).tolist(),
            conditional_accuracy_percent=100*float(v.sum(1).mean())/r['oracle_hits'],
            common_loss=float(np.mean([r['common_losses'][f'{a}_s{s}'] for s in [1,2,3]])),
            by_source={source:100*float(v[:,[i for i,e in enumerate(entries) if e['source']==source]].mean()) for source in sorted({e['source'] for e in entries})}) for a,v in values.items()}
        diagnostics={}
        for a in ARMS:
            aligned=[x for s in [1,2,3] for x in r['global_margin_aligned'][f'{a}_s{s}']]
            wrong=[x for s in [1,2,3] for x in r['global_margin_wrong_query'][f'{a}_s{s}']]
            shuffled=np.array([r['shuffled_hit_vectors'][f'{a}_s{s}'] for s in [1,2,3]],float)
            diagnostics[a]=dict(global_positive_margin=float(np.mean(aligned)),wrong_query_positive_margin=float(np.mean(wrong)),
                aligned_minus_wrong_margin=float(np.mean(np.array(aligned)-wrong)),margin_observations_per_seed=len(aligned)//3,
                shuffled_accuracy_percent=100*float(shuffled.mean()),shuffle_minus_aligned_pp=100*float((shuffled-values[a]).mean()),
                changed_predictions_per_seed=[r['changed_predictions'][f'{a}_s{s}'] for s in [1,2,3]])
        comparisons=bootstrap(values,entries,.9875 if cohort=='new_development' else .95)
        interaction=100*float((values['paired_regularized']-values['paired']-values['regularized']+values['matching']).mean())
        results[cohort]=dict(n=len(entries),oracle_hits=r['oracle_hits'],methods=methods,comparisons=comparisons,
                             diagnostics=diagnostics,factorial_interaction_pp=interaction)
        if cohort=='new_development':results[cohort]['source_group_sensitivity95']=bootstrap(values,entries,.95,'source_group')
        lines+=['',f'{cohort}: n={len(entries)}, 候选可达{r["oracle_hits"]}']
        for a,v in methods.items():lines.append(f'{a}: {v["accuracy_percent"]:.2f}%, 三种子{v["seed_hits"]}; 可达条件{v["conditional_accuracy_percent"]:.2f}%; 共同损失{v["common_loss"]:.4f}')
        for name,v in comparisons.items():lines.append(f'{name}: {v["delta_pp"]:+.2f}pp, {v["interval_percent"]:.2f}%区间{v["ci_pp"]}')
    new=results['new_development']['comparisons'];old=results['old_val']['comparisons']
    gates={k:bool(v['delta_pp']>=2 and v['ci_pp'][0]>0) for k,v in new.items()}
    advance=bool(gates['combined_vs_original'] and gates['combined_vs_matching'] and old['combined_vs_original']['delta_pp']>=-2)
    gaps={a:{c:results['train']['methods'][a]['conditional_accuracy_percent']-results[c]['methods'][a]['conditional_accuracy_percent'] for c in ['old_val','new_development']} for a in ['original']+ARMS}
    lines+=['','登记正向门槛：'+json.dumps(gates,ensure_ascii=False),f'进入新独立测试的开发候选门槛={advance}',
            '条件命中训练—开发差距：'+json.dumps(gaps,ensure_ascii=False),
            '错误指令只替换新增分支的全局query向量；原评分器、候选及原缓存图像表示仍由真实指令产生，不能当作完整指令泛化评估。',
            '配对对比可能含语义近义假负例；正例原型使用训练标签，推理前向不读标签。防记忆为dropout+锚定联合约束。',
            '旧172/新300均为重复使用的开发集，不能据此确认跨数据集优越性或超过MVP。原权重未修改，fresh280未读取，无HTML。']
    write(OUT/'results.json',dict(complete=True,cohorts=results,primary_positive_gates=gates,advance_to_new_independent_test=advance,
        conditional_train_development_gaps_pp=gaps,mainline_unchanged=True,independent_confirmation=False))
    (OUT/'results.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    # Plain scientific figure; evidence is also fully available as text/JSON.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4.2));colors=['#555555','#6b9ac4','#50a98e','#daab50','#c56e82']
    labels=['Original','Match','+Pair','+Reg','+Both']
    for a,label,color in zip(['original']+ARMS,labels,colors):
        axes[0].plot(range(3),[results[c]['methods'][a]['accuracy_percent'] for c in COHORTS],marker='o',label=label,color=color)
    axes[0].set_xticks(range(3),['Train 2973','Old dev 172','New dev 300']);axes[0].set_ylabel('Accuracy (%)');axes[0].legend(fontsize=8);axes[0].grid(alpha=.2)
    vs=list(new.values());delta=[v['delta_pp'] for v in vs];lo=[v['ci_pp'][0] for v in vs];hi=[v['ci_pp'][1] for v in vs]
    axes[1].errorbar(delta,range(4),xerr=[np.array(delta)-lo,np.array(hi)-delta],fmt='o',capsize=4)
    axes[1].set_yticks(range(4),['Both - Original','Both - Match','Pair - Match','Reg - Match']);axes[1].invert_yaxis();axes[1].axvline(0,color='gray',lw=1);axes[1].set_xlabel('New dev difference (pp), 98.75% cluster CI');axes[1].grid(alpha=.2)
    fig.suptitle('V20: fixed data / pools / original scorer; 3-fit means, reused development',fontsize=11)
    fig.tight_layout();fig.savefig(OUT/'summary.png',dpi=170);plt.close(fig)
    files=[OUT/p for p in ['protocol.json','cache/frozen.json','models/frozen.json','results.json','results.txt','summary.png']]
    for r in read(prior.MAIN/'models/frozen.json')['models'].values():
        assert sha(r['path'])==r['sha256']
    verify()
    write(OUT/'completion.json',dict(complete=True,completed_at_utc=now(),artifact_sha256={str(p):sha(p) for p in files},
         checkpoint_hashes_verified=12,original_hit_replay=10335,mainline_unchanged=True))
    registry=read(ROOT/'CURRENT_RESEARCH.json');registry['active_experiment'].update(status='complete',phase='training_evaluation_analysis_complete',
        results_path=str(OUT/'results.txt'),completion_path=str(OUT/'completion.json'),completed_at_utc=now(),
        primary_positive_gates=gates,advance_to_new_independent_test=advance)
    completed=registry.setdefault('completed_experiments',[])
    assert isinstance(completed,list)
    completed[:]=[r for r in completed if r.get('run_id')!=OUT.name]
    completed.append(dict(run_id=OUT.name,status='complete',protocol_path=str(OUT/'protocol.json'),
                         results_path=str(OUT/'results.txt'),completion_path=str(OUT/'completion.json'),mainline_unchanged=True))
    write(ROOT/'CURRENT_RESEARCH.json',registry)
    status('complete',primary_positive_gates=gates,advance_to_new_independent_test=advance);print('\n'.join(lines),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['prepare','compact','worker','train','evaluate','analyze','all'],default='all');parser.add_argument('--worker',type=int,choices=[0,1]);a=parser.parse_args()
    torch.set_num_threads(2)
    try:
        prepare()
        if a.stage in ['compact','all']:compact()
        if a.stage=='worker':worker_run(a.worker)
        if a.stage in ['train','all']:train_all()
        if a.stage in ['evaluate','all']:evaluate()
        if a.stage in ['analyze','all']:analyze()
    except BaseException:
        status('failed',worker=a.worker if a.stage=='worker' else None,error=traceback.format_exc());raise


if __name__=='__main__':main()
