"""Frozen MVP + matched-budget candidate verification pilot. No HTML/API calls."""
import os
os.environ.setdefault('PYTHONDONTWRITEBYTECODE','1')
os.environ.setdefault('HF_HUB_OFFLINE','1')
os.environ.setdefault('TRANSFORMERS_OFFLINE','1')
os.environ.setdefault('OMP_NUM_THREADS','4')
os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
import argparse
from collections import defaultdict
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import numpy as np
from PIL import Image,ImageDraw
from clickagent.mvp_repair_v21 import (PUBLIC_KEYS,CHOICE_PROMPT,stable_key,make_pools,
                                      presentation_order,visual_inputs,combine_orders)

OUT=ROOT/'runs/mvp_repair_v21'
MAIN=ROOT/'runs/quality_readout_v15_3k'
PRIOR=ROOT/'runs/bottleneck_v19'
MODEL=ROOT/'models/Qwen3-VL-8B-Instruct'
CONTRASTS={'native_verification':('native_verify','mvp'),
           'our_candidate_value':('augmented_verify','native_verify'),
           'hybrid_net_gain':('augmented_verify','mvp')}
SOURCES=[Path(__file__),ROOT/'src/clickagent/mvp_repair_v21.py',ROOT/'tests/test_mvp_repair_v21.py']


def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def now():return datetime.now(timezone.utc).isoformat()
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()
def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    for i in range(40):
        try:os.replace(tmp,p);return
        except PermissionError:
            if i==39:raise
            time.sleep(.1)
def status(phase,worker=False,**fields):
    r=dict(phase=phase,complete=phase=='complete',updated_at_utc=now(),**fields)
    write(OUT/('worker_status.json' if worker else 'status.json'),r);print(json.dumps(r,ensure_ascii=False),flush=True)


def verify():
    p=read(OUT/'protocol.json')
    for path,h in p['hashes'].items():assert sha(path)==h,path
    for path,r in p['backbone_files'].items():
        st=Path(path).stat();assert st.st_size==r['bytes'] and st.st_mtime_ns==r['mtime_ns'],path
    return p


def prepare():
    if (OUT/'protocol.json').exists():return verify()
    smoke_receipt=read(OUT/'smoke/receipt.json')
    assert smoke_receipt['complete'],'Synthetic model smoke required'
    assert smoke_receipt['source_sha256']=={str(p):sha(p) for p in SOURCES},'Source changed after smoke'
    indices=read(PRIOR/'data/indices.json');lookup={}
    source_manifests=[ROOT/'runs/multiregion_v13/data/val.jsonl',MAIN/'data/new_reserve.jsonl']
    for path in source_manifests:
        for line in path.read_text(encoding='utf-8-sig').splitlines():
            c=json.loads(line);lookup[c['sample_id']]=c
    selected=[]
    for cohort,count in [('new_development',24),('old_val',7)]:
        sources=defaultdict(lambda:defaultdict(list))
        for e in indices[cohort]:sources[e['source']][e['group']].append(e)
        for source,groups in sorted(sources.items()):
            chosen=sorted(groups,key=lambda g:stable_key(g+'|v21_rgb_20262101'))[:count];assert len(chosen)==count
            for group in chosen:
                e=min(groups[group],key=lambda e:stable_key(e['sample_id']+'|v21_query'))
                selected.append((cohort,e))
    selected.sort(key=lambda row:stable_key(row[1]['sample_id']+'|v21_execution'))
    assert len(selected)==100 and len({e['group'] for _,e in selected})==100
    assert not {e['group'] for _,e in selected}&{e['group'] for e in indices['train']}
    cases=[];labels={};proposals={};provenance={}
    adopted=read(MAIN/'models/frozen.json')['models']
    for cohort,e in selected:
        c=lookup[e['sample_id']];assert c['dataset']==e['source']
        public={k:c[k] for k in ['sample_id','dataset','image','image_size','image_sha256','instruction']}
        public.update(cohort=cohort,source=e['source'],group=e['group'],source_group=e['source_group']);cases.append(public)
        labels[c['sample_id']]=c['target_geometry']
        assert sha(e['path'])==e['sha256'];provenance[e['path']]=e['sha256']
        with np.load(e['path'],allow_pickle=False) as z:points=z['points'].copy()
        assert len(points)==e['n'];scores=[];original=[]
        for s in [1,2,3]:
            split='old_val' if cohort=='old_val' else 'new_reserve'
            path=MAIN/f'predictions/{split}/expanded_s{s}/{Path(e["path"]).stem}.npz'
            with np.load(path,allow_pickle=False) as z:
                assert str(z['checkpoint_sha256'])==adopted[f'expanded_s{s}']['sha256'] and str(z['base_sha256'])==e['sha256']
                score=z['scores'].copy()
            assert score.shape==(len(points),) and np.isfinite(score).all()
            scores.append(score.tolist());original.append(points[int(score.argmax())].tolist() if len(points) else None)
            provenance[str(path)]=sha(path)
        proposals[c['sample_id']]=dict(points=points.tolist(),scores=scores,original_seed_points=original)
    write(OUT/'data/public.json',cases);write(OUT/'data/labels.json',labels);write(OUT/'data/proposals.json',proposals)
    write(OUT/'data/provenance.json',provenance)
    timing=[]
    for source in sorted({c['source'] for c in cases}):
        timing.append(min([c for c in cases if c['source']==source],key=lambda c:stable_key(c['sample_id']+'|v21_timing'))['sample_id'])
    write(OUT/'data/timing_panel.json',dict(sample_ids=timing,n=7,warmups=2))
    identity=read(ROOT/'data/conditioned_v3/qwen_weight_identity.json');backbone={}
    for name,r in identity['files'].items():
        path=MODEL/name
        if name.endswith('.safetensors'):status('authenticating_frozen_backbone',file=name)
        assert path.stat().st_size==r['bytes'] and sha(path)==r['sha256'],path
        st=path.stat();backbone[str(path)]=dict(sha256=r['sha256'],bytes=st.st_size,mtime_ns=st.st_mtime_ns)
    inputs=SOURCES+source_manifests+[PRIOR/'data/indices.json',MAIN/'models/frozen.json',
        ROOT/'src/clickagent/advantage_runtimes.py',ROOT/'src/clickagent/public_evaluation.py',
        ROOT/'runs/crossfit_candidates_v12/runtime_identity.json',ROOT/'data/conditioned_v3/qwen_weight_identity.json',
        OUT/'smoke/receipt.json',OUT/'preflight.json']
    inputs += list((ROOT/'third_party/MVP').rglob('*.py'))
    inputs += [OUT/f'data/{name}.json' for name in ['public','labels','proposals','provenance','timing_panel']]
    inputs += [Path(adopted[f'expanded_s{s}']['path']) for s in [1,2,3]]
    inputs += [PRIOR/f'evaluation/{c}.json' for c in ['old_val','new_development']]
    spec=dict(registered_at_utc=now(),hashes={str(p):sha(p) for p in inputs},backbone_files=backbone,
        cohort='100 reused development screenshots, one query/RGB: new72(24/source),old28(7/source). SHA-based sampling independent of labels/outcomes. No fresh280 access.',
        arms=dict(original='Read-only adopted3k full-pool predictions,3 fits,not ensemble',mvp='Pinned official Qwen3VL8B,layer24,2 local views,262144/view,consistency14',
            native_verify='MVP final point plus up to2 other distinct native points, fixed frozen8B visual selector',
            augmented_verify='Keep MVP final; replace other slots with original-scorer-ranked points, one near refinement allowed and next14px diverse preferred; native fill if needed. Each original3k seed separately,mean not ensemble'),
        candidate_budget='For every case all verification arms have exactly k=min(3,unique valid native predictions), same MVP anchor. Invalid MVP => all hybrids retainNone,no verification; denominator retained.',
        verification='Fixed512x512 letterboxed full overview + one512x512 marked source512-pixel crop per candidate; identical prompt and image counts/grids across arms. Constrained first-answer-token label logits A/B/C/N,not free generated coordinates. Two deterministic hash-permuted/reversed orders averaged after label log-softmax alignment; N falls back to MVP in every arm. No learned confidence threshold.',
        budget='Each verification order uses(k+1)*262144 pixels and1 backbone prefill;2 orders. This is incremental compute over MVP. Augmented pipeline additionally needs our4-view proposal/readout cost. Accuracy caches are not latency measurements.',
        cache='Deduplicate identical candidate pools within a case across3 fits and native/augmented arms; costs report the logical single-fit2-order policy,not sum over all3 fits or cache-read latency.',
        frozen='No fitting,prompt search,seed selection,or automatic adoption. Original weights and all candidate rules fixed before any development MVP/selector output.',
        primary=dict(cohort='new72, historically reused development',contrasts=CONTRASTS,
            intervals='5000 source-stratified RGB-cluster bootstrap over fixed3-fit means;98.333% Bonferroni3 intervals. MVP/native are shared frozen policies,not3 independent fits.',
            advancement='Both our_candidate_value and hybrid_net_gain >=3pp with adjusted lower>0; old28 hybrid-minusMVP point>=-3pp. Pilot only; positive needs new independent test.'),
        secondary='Old28/all100 descriptive95%,source results,pool coverage,conditional selection,wrong repaired/right broken,order disagreement,N fallbacks.',
        timing='Record every new MVP/verification call actual time,pixels,forwards. Additionally fresh original S1 proposal pipeline on7 source-balanced screenshots after2 warmups; derive full-pool score from existing forward hook and report cache point parity. Sum module times only as an additive estimate,not jointly measured end-to-end latency. Model load excluded,no speed superiority claim.',
        interpretation='Small pilot has low power for small gains. Extra candidate value isolated against same-budget native verification; extra verification compute is controlled. Different candidate content and source processing still declared.',
        execution='Serial local worker using both P5000s for one frozen8B; resume per MVP case and per verification pool/order. Then isolated fresh-proposal timing process and CPU analysis. Python logs/JSON/TXT/PNG,no HTML.')
    for p in SOURCES:
        dest=OUT/'source_archive'/p.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest)
    write(OUT/'protocol.json',spec);write(OUT/'registry_before.json',read(ROOT/'CURRENT_RESEARCH.json'))
    reg=read(ROOT/'CURRENT_RESEARCH.json');reg['active_experiment']=dict(run_id=OUT.name,status='registered',protocol_path=str(OUT/'protocol.json'),
         status_path=str(OUT/'status.json'),mainline_unchanged=True,n=100,training=False,output_format='Python/JSON/TXT/PNG,no HTML')
    write(ROOT/'CURRENT_RESEARCH.json',reg)
    write(ROOT/'research/decisions/20261108_mvp_repair_v21.json',dict(user_authorization='可以，试试吧',registered_at_utc=now(),protocol_sha256=sha(OUT/'protocol.json'),
          objective='Matched-budget MVP candidate supplementation and visual verification; fixed original3k and native control.',mainline_unchanged=True))
    status('registered',screenshots=100,new=72,old=28)
    return spec


def sync(torch):
    for d in range(torch.cuda.device_count()):torch.cuda.synchronize(d)


def load_runtime(folder):
    from clickagent.advantage_runtimes import configure_paths
    configure_paths('mvp')
    import torch
    from clickagent.advantage_runtimes import MVPAdapter
    torch.set_num_threads(4);torch.manual_seed(20262101)
    runtime=MVPAdapter(folder,max_pixels=262144)
    return runtime,torch


def verify_once(runtime,torch,image,instruction,pool,order,preview=None):
    sync(torch);before=runtime.counter.snapshot();started=time.perf_counter()
    images,messages,geometry=visual_inputs(image,instruction,pool,order)
    if preview is not None:
        preview=Path(preview);preview.parent.mkdir(parents=True,exist_ok=True)
        canvas=Image.new('RGB',(512*len(images),512),'white')
        for i,im in enumerate(images):canvas.paste(im,(512*i,0))
        canvas.save(preview)
    text=runtime.processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
    inputs=runtime.processor(text=[text],images=images,return_tensors='pt',padding=False).to(runtime.device)
    grids=inputs['image_grid_thw'].detach().cpu().numpy()
    assert grids.tolist()==[[1,32,32]]*len(images),grids
    runtime.model.clear_captured_states()
    with torch.inference_mode():
        output=runtime.model(**inputs,use_cache=False,logits_to_keep=1,output_attentions=False,output_hidden_states=False,return_dict=True)
        logits=output.logits[0,-1].detach().float().cpu();del output
    token_ids=[]
    for label in geometry['labels']:
        ids=set()
        for prefix in ['', ' ']:
            encoded=runtime.processor.tokenizer.encode(prefix+label,add_special_tokens=False)
            if len(encoded)==1:ids.add(encoded[0])
        assert ids;token_ids.append(sorted(ids))
    assert len({i for v in token_ids for i in v})==sum(map(len,token_ids))
    scores=torch.stack([torch.logsumexp(logits[ids],0) for ids in token_ids]);assert torch.isfinite(scores).all()
    mass=float(torch.exp(torch.logsumexp(scores,0)-torch.logsumexp(logits,0)))
    sync(torch);after=runtime.counter.snapshot()
    result=dict(order=order,scores=scores.tolist(),valid_label_mass=mass,token_ids=token_ids,geometry=geometry,
        elapsed_seconds=time.perf_counter()-started,counter={k:after[k]-before[k] for k in before},
        image_grids=grids.tolist(),ground_truth_in_model_inputs=False)
    assert result['counter']['backbone_forward_calls']==1
    assert result['counter']['processed_pixels']==geometry['input_pixels']
    for im in images:im.close()
    return result


def smoke():
    OUT.mkdir(parents=True,exist_ok=True);status('synthetic_smoke_loading')
    runtime,torch=load_runtime(OUT/'smoke/runtime')
    image=Image.new('RGB',(1200,800),'white');draw=ImageDraw.Draw(image)
    draw.rectangle((120,250,380,450),fill=(220,20,20));draw.rectangle((700,250,960,450),fill=(20,70,220))
    instruction='Click the blue rectangle on the right.';pool=[[250,350],[830,350],[600,650]]
    before=runtime.counter.snapshot();sync(torch);t=time.perf_counter();mvp=runtime.infer(image,instruction);sync(torch);mvp_elapsed=time.perf_counter()-t
    observations=[verify_once(runtime,torch,image,instruction,pool,presentation_order('synthetic',3,reverse),
                   OUT/f'smoke/order_{int(reverse)}.png') for reverse in [False,True]]
    combined=combine_orders(pool,pool[0],observations)
    write(OUT/'smoke/receipt.json',dict(complete=True,at_utc=now(),source_sha256={str(p):sha(p) for p in SOURCES},
         mvp_output=mvp,mvp_seconds=mvp_elapsed,observations=observations,combined=combined,
         runtime_metadata=runtime.metadata,synthetic_only=True,semantic_smoke_success=combined['point']==pool[1]))
    image.close();status('synthetic_smoke_complete',selected_point=combined['point'],semantic_smoke_success=combined['point']==pool[1])


def worker():
    verify();cases=read(OUT/'data/public.json');proposals=read(OUT/'data/proposals.json');protocol=sha(OUT/'protocol.json');runtime=torch=None
    for i,c in enumerate(cases,1):
        dest=OUT/f'cases/{stable_key(c["sample_id"])}.json'
        if dest.exists():assert read(dest)['protocol_sha256']==protocol;continue
        if runtime is None:
            status('loading_frozen_mvp_and_selector',worker=True,total=len(cases));runtime,torch=load_runtime(OUT/'runtime');write(OUT/'runtime/metadata.json',runtime.metadata)
        assert sha(c['image'])==c['image_sha256']
        with Image.open(c['image']) as opened:image=opened.convert('RGB')
        assert list(image.size)==c['image_size']
        mpath=OUT/f'mvp/{stable_key(c["sample_id"])}.json'
        if mpath.exists():
            saved=read(mpath);assert saved['protocol_sha256']==protocol and saved['image_sha256']==c['image_sha256']
        else:
            status('mvp_inference',worker=True,completed=i-1,total=len(cases),sample_id=c['sample_id'])
            before=runtime.counter.snapshot();sync(torch);t=time.perf_counter();output=runtime.infer(image,c['instruction']);sync(torch);after=runtime.counter.snapshot()
            saved=dict(output=output,elapsed_seconds=time.perf_counter()-t,counter={k:after[k]-before[k] for k in before},
                       image_sha256=c['image_sha256'],protocol_sha256=protocol)
            write(mpath,saved)
        output=saved['output'];p=proposals[c['sample_id']]
        pools=[make_pools(output,p['points'],score,c['image_size']) for score in p['scores']]
        native=pools[0]['native'];assert all(q['native']==native for q in pools)
        if not native:
            native_result=dict(point=output.get('point'),fallback_to_mvp=True,order_disagrees=False,verification_paths=[])
            augmented=[dict(native_result) for _ in [1,2,3]]
        else:
            unique={digest(pool):pool for pool in [native]+[q['augmented'] for q in pools]};verified={}
            for signature,pool in sorted(unique.items()):
                observations=[];paths=[]
                for reverse in [False,True]:
                    path=OUT/f'verification/{stable_key(c["sample_id"])}_{signature[:16]}_{int(reverse)}.json'
                    order=presentation_order(c['sample_id'],len(pool),reverse)
                    if path.exists():
                        obs=read(path);assert obs['protocol_sha256']==protocol and obs['pool']==pool and obs['image_sha256']==c['image_sha256']
                    else:
                        status('visual_verification',worker=True,completed=i-1,total=len(cases),sample_id=c['sample_id'],pool=signature[:12],reverse=reverse)
                        preview=OUT/f'preview/{signature[:10]}_{int(reverse)}.png' if i<=1 else None
                        obs=verify_once(runtime,torch,image,c['instruction'],pool,order,preview)
                        obs.update(pool=pool,image_sha256=c['image_sha256'],protocol_sha256=protocol);write(path,obs)
                    observations.append(obs);paths.append(str(path))
                result=combine_orders(pool,pool[0],observations);result['verification_paths']=paths;verified[signature]=result
            native_result=verified[digest(native)];augmented=[verified[digest(q['augmented'])] for q in pools]
        row=dict(sample_id=c['sample_id'],protocol_sha256=protocol,image_sha256=c['image_sha256'],mvp_path=str(mpath),
            mvp_point=output.get('point'),original_seed_points=p['original_seed_points'],native_pool=native,augmented_pools=[q['augmented'] for q in pools],
            pool_details=pools,native_result=native_result,augmented_results=augmented,ground_truth_in_model_inputs=False)
        write(dest,row);image.close();status('case_complete',worker=True,completed=i,total=len(cases),candidate_count=len(native))
    files=[OUT/f'cases/{stable_key(c["sample_id"])}.json' for c in cases]
    write(OUT/'predictions_frozen.json',dict(complete=True,frozen_at_utc=now(),cases={str(p):sha(p) for p in files},protocol_sha256=protocol))
    verify();status('complete',worker=True,screenshots=len(cases))


def timing():
    verify();panel=read(OUT/'data/timing_panel.json');cases={c['sample_id']:c for c in read(OUT/'data/public.json')};proposals=read(OUT/'data/proposals.json')
    from clickagent.advantage_runtimes import configure_paths
    configure_paths('ours')
    import torch
    from clickagent.advantage_runtimes import OurRuntime
    torch.set_num_threads(4);status('loading_fresh_proposal_timing',worker=True,total=7)
    runtime=OurRuntime(OUT/'proposal_timing/runtime',seeds=(1,));captured=[]
    handle=runtime.models[0][0].register_forward_hook(lambda module,args,output:captured.append(output.detach()))
    rows=[]
    for j,sid in enumerate(panel['sample_ids'][:2]+panel['sample_ids']):
        c=cases[sid];captured.clear()
        with Image.open(c['image']) as opened:image=opened.convert('RGB')
        sync(torch);before=runtime.counter.snapshot();t=time.perf_counter();runtime.infer(image,c['instruction']);sync(torch)
        elapsed=time.perf_counter()-t;after=runtime.counter.snapshot();r=runtime.last_record
        point=r['points'][int(captured[-1].argmax())].tolist() if r['n'] else None
        image.close()
        if j<2:continue
        rows.append(dict(sample_id=sid,elapsed_seconds=elapsed,counter={k:after[k]-before[k] for k in before},point=point,
                         cached_point=proposals[sid]['original_seed_points'][0],point_parity=point==proposals[sid]['original_seed_points'][0],
                         score_head_forward_count=len(captured),warmup=False))
        status('fresh_proposal_timing',worker=True,completed=len(rows),total=7)
    handle.remove();write(OUT/'proposal_timing/results.json',dict(complete=True,rows=rows,warmups=2,
        implementation='Original OurRuntime computes full logits but returnsK16; hook reads existing logits to recover full-union top1 without extra model pass.',
        cache_not_used=True,original_scorer_seed=1,model_loading_excluded=True))
    verify();status('complete',worker=True,timing_cases=7)


def run_child(stage):
    stdout_path=OUT/f'{stage}.stdout.log';stderr_path=OUT/f'{stage}.stderr.log'
    with stdout_path.open('a',encoding='utf-8') as stdout,stderr_path.open('a',encoding='utf-8') as stderr:
        process=subprocess.Popen([sys.executable,'-B','-X','utf8','-u',str(Path(__file__).resolve()),'--stage',stage],cwd=ROOT,stdout=stdout,stderr=stderr)
    write(OUT/f'{stage}.execution.json',dict(pid=process.pid,started_at_utc=now()))
    while process.poll() is None:
        progress=read(OUT/'worker_status.json') if (OUT/'worker_status.json').exists() else {}
        status(stage+'_running',worker_progress=progress);time.sleep(50)
    assert process.returncode==0,f'{stage} failed; inspect {stderr_path}'


def bootstrap(values,cases,level):
    groups=defaultdict(lambda:defaultdict(list))
    for i,c in enumerate(cases):groups[c['source']][c['group']].append(i)
    strata=[list(x.values()) for x in groups.values()];rng=np.random.default_rng(20262102)
    differences={k:(values[a]-values[b]).mean(0) for k,(a,b) in CONTRASTS.items()};draws={k:[] for k in differences}
    for _ in range(5000):
        ix=[i for clusters in strata for j in rng.integers(0,len(clusters),len(clusters)) for i in clusters[j]]
        for k,d in differences.items():draws[k].append(d[ix].mean())
    alpha=1-level
    return {k:dict(delta_pp=100*float(d.mean()),ci_pp=(100*np.quantile(draws[k],[alpha/2,1-alpha/2])).tolist(),
                   interval_percent=100*level,seed_deltas_pp=(100*(values[CONTRASTS[k][0]]-values[CONTRASTS[k][1]]).mean(1)).tolist()) for k,d in differences.items()}


def analyze():
    verify();freeze=read(OUT/'predictions_frozen.json');assert freeze['complete']
    for p,h in freeze['cases'].items():assert sha(p)==h
    cases=read(OUT/'data/public.json');labels=read(OUT/'data/labels.json');proposals=read(OUT/'data/proposals.json')
    from clickagent.public_evaluation import point_in_geometry
    def hit(p,c):return bool(p is not None and point_in_geometry([p[i]/c['image_size'][i] for i in [0,1]],labels[c['sample_id']]))
    vectors={a:[[],[],[]] for a in ['original','mvp','native_verify','augmented_verify']};oracles={a:[[],[],[]] for a in ['original','native_verify','augmented_verify']}
    costs=[];rows=[];replay=0
    references={c:read(PRIOR/f'evaluation/{c}.json') for c in ['old_val','new_development']}
    reference_index={c:{sid:i for i,sid in enumerate(v['sample_ids'])} for c,v in references.items()}
    def vcost(result):
        observations=[read(p) for p in result['verification_paths']]
        return dict(elapsed_seconds=sum(x['elapsed_seconds'] for x in observations),
                    backbone_forward_calls=sum(x['counter']['backbone_forward_calls'] for x in observations),
                    processed_pixels=sum(x['counter']['processed_pixels'] for x in observations))
    for c in cases:
        r=read(OUT/f'cases/{stable_key(c["sample_id"])}.json');assert not r['ground_truth_in_model_inputs']
        mvp=hit(r['mvp_point'],c);native=hit(r['native_result']['point'],c)
        for s in range(3):
            orig=hit(r['original_seed_points'][s],c);aug=hit(r['augmented_results'][s]['point'],c)
            for a,h in [('original',orig),('mvp',mvp),('native_verify',native),('augmented_verify',aug)]:vectors[a][s].append(h)
            for a,pool in [('original',proposals[c['sample_id']]['points']),('native_verify',r['native_pool']),('augmented_verify',r['augmented_pools'][s])]:oracles[a][s].append(any(hit(p,c) for p in pool))
            ref=references[c['cohort']]['hit_vectors'][f'original_s{s+1}'][reference_index[c['cohort']][c['sample_id']]]
            assert orig==ref;replay+=1
            assert len(r['native_pool'])==len(r['augmented_pools'][s])
        native_cost=vcost(r['native_result']);aug_cost=[vcost(x) for x in r['augmented_results']]
        assert all(x['backbone_forward_calls']==native_cost['backbone_forward_calls'] and x['processed_pixels']==native_cost['processed_pixels'] for x in aug_cost)
        m=read(r['mvp_path']);costs.append(dict(sample_id=c['sample_id'],mvp=dict(elapsed_seconds=m['elapsed_seconds'],**m['counter']),
            native_verification=native_cost,augmented_verification_seed_mean={k:float(np.mean([x[k] for x in aug_cost])) for k in native_cost}))
        rows.append(r)
    values={a:np.asarray(v,float) for a,v in vectors.items()};oracle_values={a:np.asarray(v,float) for a,v in oracles.items()};results={}
    lines=['V21 固定MVP + 候选补充/视觉复核：100张复用开发截图，非新独立测试。',
           '不训练、不选种子、不改原3k权重。新增72/旧28，每RGB只一个查询。',
           'native/augmented逐样本候选数、MVP保留点、提示词、图像网格与两次顺序复核预算一致。',
           'augmented分别使用原版三个种子的候选，报告均值，不把三个种子集成。']
    for cohort in ['new_development','old_val','all']:
        ix=[i for i,c in enumerate(cases) if cohort=='all' or c['cohort']==cohort];subset=[cases[i] for i in ix];vv={a:v[:,ix] for a,v in values.items()}
        methods={}
        for a,v in vv.items():
            methods[a]=dict(accuracy_percent=100*float(v.mean()),seed_hits=v.sum(1).astype(int).tolist(),
                by_source={source:100*float(v[:,[j for j,c in enumerate(subset) if c['source']==source]].mean()) for source in sorted({c['source'] for c in subset})})
            if a!='original':
                base=vv['mvp'];methods[a].update(repaired_seed_counts=((v>base).sum(1)).astype(int).tolist(),broken_seed_counts=((v<base).sum(1)).astype(int).tolist())
            if a in oracle_values:
                oo=oracle_values[a][:,ix];methods[a].update(pool_coverage_percent=100*float(oo.mean()),oracle_seed_hits=oo.sum(1).astype(int).tolist(),
                    conditional_selection_percent=100*float(v.sum())/oo.sum() if oo.sum() else None)
        comparisons=bootstrap(vv,subset,1-.05/3 if cohort=='new_development' else .95)
        results[cohort]=dict(n=len(ix),methods=methods,comparisons=comparisons)
        lines+=['',f'{cohort}: n={len(ix)}']
        for a,r in methods.items():lines.append(f'{a}: {r["accuracy_percent"]:.2f}%, 命中{r["seed_hits"]}; 修正{r.get("repaired_seed_counts")},破坏{r.get("broken_seed_counts")}; 候选覆盖{r.get("pool_coverage_percent")}')
        for k,r in comparisons.items():lines.append(f'{k}: {r["delta_pp"]:+.2f}pp,{r["interval_percent"]:.3f}%区间{r["ci_pp"]}')
    new=results['new_development']['comparisons'];old=results['old_val']['comparisons']
    gates={k:bool(r['delta_pp']>=3 and r['ci_pp'][0]>0) for k,r in new.items()}
    advance=bool(gates['our_candidate_value'] and gates['hybrid_net_gain'] and old['hybrid_net_gain']['delta_pp']>=-3)
    timing_result=read(OUT/'proposal_timing/results.json');assert timing_result['complete']
    lookup_cost={r['sample_id']:r for r in costs};panel=[]
    for t in timing_result['rows']:
        r=lookup_cost[t['sample_id']];panel.append(dict(sample_id=t['sample_id'],mvp=r['mvp']['elapsed_seconds'],
            proposal_s1=t['elapsed_seconds'],native_verification=r['native_verification']['elapsed_seconds'],
            augmented_verification=r['augmented_verification_seed_mean']['elapsed_seconds'],
            additive_hybrid_estimate=r['mvp']['elapsed_seconds']+t['elapsed_seconds']+r['augmented_verification_seed_mean']['elapsed_seconds']))
    timing_summary={k:dict(mean_seconds=float(np.mean([r[k] for r in panel])),median_seconds=float(np.median([r[k] for r in panel]))) for k in ['mvp','proposal_s1','native_verification','augmented_verification','additive_hybrid_estimate']}
    diagnostics=dict(invalid_mvp=sum(not r['native_pool'] for r in rows),
        candidate_counts=[len(r['native_pool']) for r in rows],native_order_disagreements=sum(r['native_result']['order_disagrees'] for r in rows),
        augmented_order_disagreements_per_seed=[sum(r['augmented_results'][s]['order_disagrees'] for r in rows) for s in range(3)],
        native_N_fallbacks=sum(r['native_result']['fallback_to_mvp'] for r in rows),
        augmented_N_fallbacks_per_seed=[sum(r['augmented_results'][s]['fallback_to_mvp'] for r in rows) for s in range(3)],
        fresh_proposal_point_parity=sum(r['point_parity'] for r in timing_result['rows']),fresh_proposal_timing_n=7)
    lines+=['','主比较门槛：'+json.dumps(gates,ensure_ascii=False),f'进入新独立验证候选门槛={advance}',
            '7张计时面板：'+json.dumps(timing_summary,ensure_ascii=False),
            '模块相加只是成本估算，不是联合端到端计时；复核增加计算，补充候选还需原提议管线。缓存读取不算速度收益。',
            '诊断：'+json.dumps(diagnostics,ensure_ascii=False),
            '小型复用开发试验，不能证明跨数据集/MVP榜单优势；未达到门槛不排除小收益或其他视觉复核方法。',
            '没有原fresh280数据读取、没有训练或自动采用、没有HTML。']
    write(OUT/'results.json',dict(complete=True,cohorts=results,primary_positive_gates=gates,advance_to_new_independent_test=advance,
        diagnostics=diagnostics,timing_panel=panel,timing_summary=timing_summary,per_case_costs=costs,hit_vectors=vectors,original_hit_replay=replay,
        independent_confirmation=False,mainline_unchanged=True))
    (OUT/'results.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4.1));names=['original','mvp','native_verify','augmented_verify'];xx=np.arange(4)
    axes[0].bar(xx-.18,[results['new_development']['methods'][a]['accuracy_percent'] for a in names],width=.36,label='New dev72')
    axes[0].bar(xx+.18,[results['old_val']['methods'][a]['accuracy_percent'] for a in names],width=.36,label='Old dev28')
    axes[0].set_xticks(xx,['Original','MVP','Native verify','+Our candidates'],rotation=12);axes[0].set_ylabel('Accuracy (%)');axes[0].legend(fontsize=8)
    for j,a in enumerate(['native_verify','augmented_verify']):
        m=results['new_development']['methods'][a];axes[1].bar(j-.17,np.mean(m['repaired_seed_counts']),.34,label='MVP errors repaired' if j==0 else None)
        axes[1].bar(j+.17,np.mean(m['broken_seed_counts']),.34,label='MVP correct broken' if j==0 else None)
    axes[1].set_xticks([0,1],['Native verify','+Our candidates']);axes[1].set_ylabel('New dev72 case count,3-fit mean');axes[1].legend(fontsize=8)
    fig.suptitle('V21: matched verification budget; reused development pilot',fontsize=11);fig.tight_layout();fig.savefig(OUT/'summary.png',dpi=160);plt.close(fig)
    verify()
    files=[OUT/p for p in ['protocol.json','predictions_frozen.json','proposal_timing/results.json','results.json','results.txt','summary.png']]
    write(OUT/'completion.json',dict(complete=True,completed_at_utc=now(),artifact_sha256={str(p):sha(p) for p in files},original_hit_replay=replay,mainline_unchanged=True))
    reg=read(ROOT/'CURRENT_RESEARCH.json');reg['active_experiment'].update(status='complete',phase='inference_timing_and_analysis_complete',
        results_path=str(OUT/'results.txt'),completion_path=str(OUT/'completion.json'),completed_at_utc=now(),primary_positive_gates=gates,advance_to_new_independent_test=advance)
    reg['completed_experiments'].append(dict(run_id=OUT.name,status='complete',results_path=str(OUT/'results.txt'),mainline_unchanged=True))
    write(ROOT/'CURRENT_RESEARCH.json',reg);status('complete',primary_positive_gates=gates,advance_to_new_independent_test=advance);print('\n'.join(lines),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['smoke','prepare','worker','timing','analyze','all'],default='all');a=parser.parse_args()
    try:
        if a.stage=='smoke':smoke();return
        prepare()
        if a.stage=='worker':worker()
        if a.stage=='timing':timing()
        if a.stage=='all':run_child('worker');run_child('timing')
        if a.stage in ['analyze','all']:analyze()
    except BaseException:
        status('failed',worker=a.stage in ['worker','timing'],stage=a.stage,error=traceback.format_exc());raise


if __name__=='__main__':main()
