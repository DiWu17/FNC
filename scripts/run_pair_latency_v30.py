"""Fresh matched-image latency of V29's default policy and pinned MVP.

No training, output cache, dashboard, or test-label input to either runtime.
Conditional execution is checked against the sealed V28/V29 development outputs
before timing: valid free coordinates need no legacy ranking or locked view.
"""
import os
for _name, _value in dict(PYTHONDONTWRITEBYTECODE='1', HF_HUB_OFFLINE='1',
        TRANSFORMERS_OFFLINE='1', OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4',
        MKL_NUM_THREADS='4').items():
    os.environ.setdefault(_name, _value)
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
from io import BytesIO
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts')]
import numpy as np
from PIL import Image
from clickagent.coordinate_calibration_v29 import features, rank_scores, choose

OUT = ROOT/'runs/pair_latency_v30'
V28 = ROOT/'runs/grounded_composition_v28'
V29 = ROOT/'runs/coordinate_calibration_v29'
NAMES = [f'{family}_s{s}' for family in ['original3k', 'full'] for s in [1, 2, 3]]

def now(): return datetime.now(timezone.utc).isoformat()
def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def key(s): return hashlib.sha256(s.encode()).hexdigest()[:24]
def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(8*1024*1024), b''): h.update(b)
    return h.hexdigest()
def default(v):
    if isinstance(v, np.ndarray): return v.tolist()
    if isinstance(v, np.generic): return v.item()
    raise TypeError(type(v).__name__)
def write(p, value):
    p = Path(p);p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=default,
                              allow_nan=False)+'\n', encoding='utf-8')
    os.replace(tmp, p)
def status(method, phase, **fields):
    value = dict(method=method, phase=phase, complete=phase=='complete',
                 pid=os.getpid(), updated_at_utc=now(), **fields)
    write(OUT/method/'status.json', value)
    print(json.dumps(value, ensure_ascii=False, default=default), flush=True)
def load_npz(p):
    with np.load(p, allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files if k!='metadata'}, json.loads(str(z['metadata']))
def load_ranker(row):
    assert sha(row['path'])==row['sha256'], row['path']
    a, m = load_npz(row['path']);return dict(m, **a)

def a32(points, scores):
    points, scores = np.asarray(points), np.asarray(scores)
    ix = []
    if len(points):
        for s in scores:
            j = int(s.argmax())
            if j not in ix: ix.append(j)
        z = (scores-scores.mean(1, keepdims=True))/np.maximum(scores.std(1, keepdims=True), 1e-6)
        for j in np.argsort(-z.mean(0), kind='stable'):
            if len(ix)>=min(32, len(points)): break
            if int(j) not in ix: ix.append(int(j))
    ix = np.asarray(sorted(ix), int)
    return points[ix], scores[:, ix], ix

def fallback_center(points, scores, size):
    if not len(points): return [size[0]//2, size[1]//2]
    z = (scores-scores.mean(1, keepdims=True))/np.maximum(scores.std(1, keepdims=True), 1e-6)
    return points[int(z.mean(0).argmax())].tolist()

def assembled_records(old_points, old_scores, native, names, crop_box, size):
    pp = np.asarray(old_points).tolist();oldn = len(pp)
    for part in ['global', 'free', 'locked']:
        p = native[part]
        if p is not None and p not in pp: pp.append(p)
    points = np.asarray(pp, int).reshape(-1, 2)
    return [features(points, oldn,
        np.r_[s, np.full(len(pp)-oldn, float(np.median(s)) if len(s) else 0.)],
        native, names, crop_box, size) for s in old_scores]

def conditional_choice(native, r, model):
    # This is an execution shortcut for choose(r, rank_scores(model,r), gate=None).
    # It does not approximate ranking or apply a new confidence threshold.
    if native['free'] is not None: return list(native['free'])
    j, _ = choose(r, rank_scores(model, r))
    return r['points'][j].tolist() if j is not None else None

def parity():
    frozen = read(V29/'frozen.json')
    models = [load_ranker(frozen['models'][n+'__source']) for n in NAMES]
    cases = [c for c in read(V28/'data/public.json') if c['cohort']!='train']
    counts = Counter()
    for c in cases:
        ident = key(c['sample_id'])
        a, am = load_npz(V28/'original'/f'{ident}.npz')
        pp, ss, ix = a32(a['complete_points'], a['complete_scores'])
        assert np.array_equal(pp, a['points']) and np.array_equal(ss, a['scores'])
        assert ix.tolist()==am['indices'];counts['A32_rebuilds']+=1
        g, gm = load_npz(V28/'grounded'/f'{ident}.npz')
        _, global_meta = load_npz(V28/'raw/global'/f'{ident}.npz')
        names = {p: gm[p+'_target'] for p in ['global', 'free', 'locked']}
        rr = assembled_records(pp, ss, gm['native_points'], names,
                               global_meta['planned_local_transform']['box'], c['image_size'])
        old, _ = load_npz(V29/'features'/f'{ident}.npz')
        for si, (r, model) in enumerate(zip(rr, models)):
            for field in ['x', 'q', 'base', 'source_base']:
                assert np.allclose(r[field], old[field][si], atol=1e-10, rtol=1e-10), (c['sample_id'], field)
            assert np.array_equal(r['points'], old['points'])
            j, _ = choose(r, rank_scores(model, r))
            expected = r['points'][j].tolist() if j is not None else None
            assert conditional_choice(gm['native_points'], r, model)==expected
            counts['six_head_point_parity']+=1
            counts['valid_free' if gm['native_points']['free'] is not None else 'fallback']+=1
    assert counts['six_head_point_parity']==600
    return dict(complete=True, cached_parity_only_not_latency=True, cases=len(cases), **counts)

def prepare():
    if (OUT/'protocol.json').exists(): return verify()
    check = parity();write(OUT/'parity.json', check)
    manifest = ROOT/'data/public_v1/manifests/screenspot_pro.jsonl'
    all_rows = [json.loads(s) for s in manifest.read_text(encoding='utf-8-sig').splitlines()]
    # Selection uses public ID, application, and UI type only, never target boxes,
    # previous success/failure, outputs, or runtimes.
    ordered = sorted(all_rows, key=lambda c: key('v30_latency_20261110|'+c['sample_id']))
    selected = [];apps = set()
    for ui in ['text', 'icon']:
        for c in ordered:
            if sum(x['metadata']['ui_type']==ui for x in selected)>=3: break
            if c['metadata']['ui_type']==ui and c['metadata']['application'] not in apps:
                selected.append(c);apps.add(c['metadata']['application'])
    assert len(selected)==6
    selected.sort(key=lambda c: key('v30_order|'+c['sample_id']))
    ids = {c['sample_id'] for c in selected}
    warm = [c for c in ordered if c['sample_id'] not in ids][:1]
    def public(c):
        return {k:c[k] for k in ['sample_id', 'dataset', 'image', 'image_size', 'instruction']} | dict(
            image_sha256=sha(c['image']), application=c['metadata']['application'], ui_type=c['metadata']['ui_type'])
    cases, warm = [public(c) for c in selected], [public(c) for c in warm]
    write(OUT/'data/public.json', cases);write(OUT/'data/warmup.json', warm)
    # This timing pilot intentionally does not score accuracy or tune on labels.
    source_paths = [Path(__file__), ROOT/'src/clickagent/coordinate_calibration_v29.py',
        ROOT/'src/clickagent/grounded_composition_v28.py', ROOT/'src/clickagent/advantage_runtimes.py',
        ROOT/'src/clickagent/conditioned_features.py', ROOT/'src/clickagent/target_token.py',
        ROOT/'src/clickagent/action_model.py', ROOT/'src/clickagent/model_features.py',
        ROOT/'scripts/run_quality_readout_v15.py', ROOT/'scripts/run_evidence_retention_v14.py',
        ROOT/'runs/crossfit_candidates_v12/runtime_identity.json', V29/'frozen.json',
        ROOT/'runs/visual_feedback_v23/protocol.json', manifest, OUT/'data/public.json', OUT/'data/warmup.json',
        OUT/'parity.json', ROOT/'runs/crossfit_candidates_v12/proposers/full.pt']
    source_paths += list((ROOT/'third_party/MVP').rglob('*.py'))
    old_models = read(ROOT/'runs/visual_feedback_v23/protocol.json')['models']
    rankers = read(V29/'frozen.json')['models']
    for row in [*old_models.values(), *[rankers[n+'__source'] for n in NAMES]]:
        assert sha(row['path'])==row['sha256'];source_paths.append(Path(row['path']))
    cfg = read(ROOT/'runs/crossfit_candidates_v12/runtime_identity.json')
    for e in cfg['actual_model_files']:
        st=Path(e['path']).stat();assert st.st_size==e['bytes'] and st.st_mtime_ns==e['mtime_ns']
    spec = dict(registered_at_utc=now(), user_request='测试一下两个方法现在单次的时间对比，为什么之前预估需要那么久',
        purpose='Fresh per-query timing pilot, not full benchmark accuracy or a speed superiority confirmation',
        cases=6, repetitions=2, warmup_cases_per_method=1, batch_size=1, max_pixels_per_view=262144,
        backbone='Same local frozen Qwen3-VL-8B-Instruct FP16 on both Quadro P5000 GPUs; separate processes sequentially',
        timing='Synchronize both GPUs. Include PNG decode, preprocessing, generation, hooks, crop construction, and scoring. Exclude model loading, file-byte read/hash, report writes, and warmup. No feature/prediction/KV cache across queries.',
        order='Ours then MVP; repetition2 reverses image order. No simultaneous backbone jobs.',
        ours='V29 keep_all_free default, learned gate disabled. Exact V28 global160/local56 prompts, layers15/28/36. Skip unused legacyA and locked observation when free valid; exact early-exit output equivalence tested on all100 cached dev cases x6 heads. If global missing generate full A to plan original crop; if free missing generate A and locked then source-calibrated fallback. Six fixed head outputs, no seed selection/ensemble. Adopted V15 mainline unchanged.',
        mvp='Pinned official 8B adapter: layer24 comma capture, max2 subimages, threshold14, SDPA/FP16. Original inference math; generation timing wrappers only.',
        fairness='Equal screenshot/instruction/backbone/per-view pixel limit/hardware, different native prompts, processor minimums and runtime/device placement retained and recorded; unequal calls/tokens/total pixels measured. No official high-resolution/default-leaderboard speed claim.',
        model_file_inventory=cfg['actual_model_files'], hashes={str(p):sha(p) for p in source_paths})
    write(OUT/'protocol.json', spec);status('controller','registered',cases=6,repetitions=2,parity=check)
    return spec

def verify():
    p = read(OUT/'protocol.json')
    for path, h in p['hashes'].items(): assert sha(path)==h, path
    return p

def sync(torch):
    for d in range(torch.cuda.device_count()): torch.cuda.synchronize(d)

class CurrentRuntime:
    def __init__(self, folder):
        import torch
        from dataclasses import replace
        from clickagent.grounded_composition_v28 import GroundedRuntime
        from clickagent.advantage_runtimes import RuntimeCounter
        from clickagent.evidence_retention import RetentionModel
        import run_quality_readout_v15 as base
        legacy = base.Extractor()
        self.base, self.proposer = base, legacy.proposer.eval().requires_grad_(False)
        self.runtime = GroundedRuntime(replace(legacy.config, offload_folder=str(folder/'offload')))
        self.heads=[]
        for name in NAMES:
            row=read(ROOT/'runs/visual_feedback_v23/protocol.json')['models'][name]
            assert sha(row['path'])==row['sha256']
            saved=torch.load(row['path'], map_location='cpu', weights_only=True)
            model=RetentionModel().eval().requires_grad_(False);model.load_state_dict(saved['state_dict'])
            self.heads.append((model,saved))
        frozen=read(V29/'frozen.json')
        self.rankers=[load_ranker(frozen['models'][n+'__source']) for n in NAMES]
        self.counter=RuntimeCounter();self.counter.attach(self.runtime.model,16)
        self.metadata=dict(method='V29_keep_all_free_conditional', runtime=self.runtime.runtime_metadata,
            policy='keep exact free, source fallback; no learned gate', all_six_fixed_heads=True,
            scheduling='Output-equivalent conditional observations; no output/evidence cache',
            model_device_map={k:str(v) for k,v in self.runtime.model.hf_device_map.items()})

    def legacy_a(self,image,instruction,stages):
        import torch
        from clickagent.evidence_retention import STATE_DIM
        from run_evidence_retention_v14 import tensors
        size=list(image.size);base=self.base
        def observe(im,part):
            started=time.perf_counter();arrays,meta=self.runtime.infer_spatial_features(im,instruction)
            stages.append(dict(part=part,elapsed_seconds=time.perf_counter()-started,
                tokens=len(meta['generated_token_ids']),input_tokens=meta['input_tokens'],
                text=meta['generated_text'],parse=meta.get('parse_status'),image_size=list(im.size)))
            return arrays,meta
        arrays,meta=observe(image,'legacy_global');views=[(arrays,meta,None)]
        global_result=base.predict_raw(self.proposer,arrays,meta,size,topk=64);local=[]
        for crop in base.plan_multiregion_crops(global_result,size):
            tr=crop['transform']
            with image.crop(tr['box']).resize(tuple(tr['crop_size']),Image.Resampling.LANCZOS) as im:
                arrays,meta=observe(im,'legacy_local_'+str(crop['region_index']))
            decoded=base.predict_raw(self.proposer,arrays,meta,tr['crop_size'],topk=64)
            views.append((arrays,meta,tr));local.append(dict(region_index=crop['region_index'],result=decoded,transform=tr))
        pipe=dict(global_result=global_result,local_views=local,pools=base.build_c1_c3_pools(global_result,local))
        b=base.build_base(pipe,size);ff=[];mm=[]
        for arrays,meta,tr in views:
            f,m=base.state_features(arrays,meta,b['points'],size,tr);ff.append(f);mm.append(m)
        tmax=max(1,*(f.shape[1] for f in ff));n=len(b['points'])
        state=np.zeros((n,4,tmax,STATE_DIM),np.float16);mask=np.zeros((n,4,tmax),bool)
        for j,(f,m) in enumerate(zip(ff,mm)):
            state[:,j,:f.shape[1]]=f;mask[:,j,:f.shape[1]]=m[:,None]
        r=dict(b,sample_id='latency_public',n=n,state=state,mask=mask);scores=[]
        with torch.inference_mode():
            for model,saved in self.heads:
                if n:
                    inputs,_=tensors([r],saved['mean'].numpy(),saved['std'].numpy(),'aligned','cpu')
                    scores.append(model(**inputs).numpy())
                else:scores.append(np.empty(0))
        pp,ss,_=a32(b['points'],np.asarray(scores))
        return pp,ss

    def infer(self,image,instruction):
        from clickagent.grounded_composition_v28 import GLOBAL_PROMPT,LOCAL_PROMPT,pixel,fixed_crop
        stages=[];size=list(image.size);old=None
        def observe(im,part,ins,prompt,cap):
            started=time.perf_counter();arrays,m=self.runtime.infer(im,ins,prompt,cap)
            stages.append(dict(part=part,elapsed_seconds=time.perf_counter()-started,
                tokens=len(m['generated_token_ids']),input_tokens=m['input_tokens'],
                image_size=list(im.size),text=m['generated_text'],complete=m['complete'],
                point=m['point'],target=m['target'],grid_shape=m['grid_shape']))
            del arrays
            return m
        gm=observe(image,'global',instruction,GLOBAL_PROMPT,160);gp=pixel(gm['point'],size)
        center=gp
        if center is None:
            old=self.legacy_a(image,instruction,stages);center=fallback_center(*old,size)
        tr=fixed_crop(center,size)
        with image.crop(tr['box']).resize(tuple(tr['crop_size']),Image.Resampling.LANCZOS) as im:
            fm=observe(im,'free',instruction,LOCAL_PROMPT,56);fp=pixel(fm['point'],list(im.size),tr)
            if fp is not None:
                return dict(point=fp,seed_points=[list(fp) for _ in NAMES],stages=stages,
                    generation_calls=len(stages),reason='keep_exact_free',crop=tr,ground_truth_in_model_input=False)
            # The locked call remains necessary for source-calibrated fallback.
            locked_instruction=instruction+'\nResolved target identity from the full screenshot: '+(gm['target'] or 'UNKNOWN')+'. Locate THIS SAME referent in this crop. Keep the original instruction constraints; use NONE if it is absent.'
            lm=observe(im,'locked',locked_instruction,LOCAL_PROMPT,56);lp=pixel(lm['point'],list(im.size),tr)
        if old is None:old=self.legacy_a(image,instruction,stages)
        native=dict(global_=gp,free=fp,locked=lp);native['global']=native.pop('global_')
        names=dict(global_=gm['target'],free=fm['target'],locked=lm['target']);names['global']=names.pop('global_')
        rr=assembled_records(*old,native,names,tr['box'],size)
        points=[conditional_choice(native,r,model) for r,model in zip(rr,self.rankers)]
        return dict(point=points[0],seed_points=points,stages=stages,generation_calls=len(stages),
            reason='free_missing_ranker_fallback',crop=tr,ground_truth_in_model_input=False)

def instrument_mvp(runtime,torch):
    original=runtime.model.generate;runtime.generation_records=[]
    def generate(*args,**kwargs):
        sync(torch);start=time.perf_counter()
        inputs=kwargs.get('input_ids');input_length=int(inputs.shape[-1]) if inputs is not None else None
        output=original(*args,**kwargs);sync(torch)
        ids=output.sequences if hasattr(output,'sequences') else output
        runtime.generation_records.append(dict(part='mvp_generate_'+str(len(runtime.generation_records)+1),
            elapsed_seconds=time.perf_counter()-start,input_tokens=input_length,
            tokens=int(ids.shape[-1])-input_length if input_length is not None else None))
        return output
    runtime.model.generate=generate

def worker(method):
    spec=verify()
    from clickagent.advantage_runtimes import configure_paths
    configure_paths('mvp' if method=='mvp' else 'coordinate')
    import torch
    torch.set_num_threads(4);folder=OUT/method;status(method,'loading_model')
    started=time.perf_counter()
    if method=='ours':runtime=CurrentRuntime(folder)
    else:
        from clickagent.advantage_runtimes import MVPAdapter
        runtime=MVPAdapter(folder,262144);instrument_mvp(runtime,torch)
    sync(torch);load_seconds=time.perf_counter()-started
    write(folder/'runtime.json',dict(load_seconds=load_seconds,metadata=runtime.metadata,
        torch_version=torch.__version__,transformers_version=__import__('transformers').__version__,
        gpus=[torch.cuda.get_device_name(d) for d in range(torch.cuda.device_count())],
        cuda_version=torch.version.cuda))
    status(method,'model_loaded',load_seconds=load_seconds)
    def once(c,rep,warm=False):
        payload=Path(c['image']).read_bytes();assert hashlib.sha256(payload).hexdigest()==c['image_sha256']
        sync(torch)
        for d in range(torch.cuda.device_count()):torch.cuda.reset_peak_memory_stats(d)
        before=runtime.counter.snapshot()
        if method=='mvp':runtime.generation_records=[]
        started=time.perf_counter()
        with Image.open(BytesIO(payload)) as source:image=source.convert('RGB')
        assert list(image.size)==c['image_size']
        try:output=runtime.infer(image,c['instruction'])
        finally:image.close()
        sync(torch);seconds=time.perf_counter()-started;after=runtime.counter.snapshot()
        assert output['ground_truth_in_model_input'] is False
        if method=='mvp':
            stages=runtime.generation_records.copy()
            output['stages']=stages;output['generation_calls']=len(stages)
        row=dict(sample_id=c['sample_id'],method=method,repetition=rep,warmup=warm,
            application=c['application'],ui_type=c['ui_type'],image_size=c['image_size'],
            elapsed_seconds=seconds,counter={k:after[k]-before[k] for k in before},
            peak_allocated_bytes=[torch.cuda.max_memory_allocated(d) for d in range(torch.cuda.device_count())],
            output=output)
        path=folder/('warmup' if warm else 'cases')/f'{key(c["sample_id"])}_r{rep}.json'
        write(path,row)
        return row
    for c in read(OUT/'data/warmup.json'):
        row=once(c,0,True);status(method,'warmup_complete',seconds=row['elapsed_seconds'])
    cases=read(OUT/'data/public.json');rows=[]
    for rep in [1,2]:
        for c in cases if rep==1 else list(reversed(cases)):
            row=once(c,rep);rows.append(row)
            status(method,'measuring',completed=len(rows),total=len(cases)*2,
                sample_id=c['sample_id'],repetition=rep,seconds=row['elapsed_seconds'],
                generation_calls=row['output']['generation_calls'],counter=row['counter'])
    write(folder/'results.json',dict(complete=True,model_load_seconds=load_seconds,rows=rows))
    status(method,'complete',measured_queries=len(rows),mean_seconds=float(np.mean([r['elapsed_seconds'] for r in rows])))

def analyze():
    verify();result={};by={}
    for method in ['ours','mvp']:
        payload=read(OUT/method/'results.json');assert payload['complete'];rows=payload['rows']
        assert len(rows)==12
        seconds=np.asarray([r['elapsed_seconds'] for r in rows]);by[method]={}
        for r in rows:by[method].setdefault(r['sample_id'],[]).append(r['elapsed_seconds'])
        stages={}
        for r in rows:
            for s in r['output']['stages']:
                stages.setdefault(s['part'],[]).append(s)
        result[method]=dict(n_screenshots=len(by[method]),repetitions=2,mean_seconds=float(seconds.mean()),
            median_seconds=float(np.median(seconds)),min_seconds=float(seconds.min()),max_seconds=float(seconds.max()),
            model_load_seconds=payload['model_load_seconds'],
            mean_generation_calls=float(np.mean([r['output']['generation_calls'] for r in rows])),
            mean_forward_calls=float(np.mean([r['counter']['backbone_forward_calls'] for r in rows])),
            mean_processed_pixels=float(np.mean([r['counter']['processed_pixels'] for r in rows])),
            mean_generated_tokens=float(np.mean([sum(s['tokens'] or 0 for s in r['output']['stages']) for r in rows])),
            fallback_queries=sum(r['output'].get('reason')=='free_missing_ranker_fallback' for r in rows),
            by_stage={name:dict(n=len(ss),mean_seconds=float(np.mean([s['elapsed_seconds'] for s in ss])),
                mean_tokens=float(np.mean([s['tokens'] or 0 for s in ss]))) for name,ss in stages.items()},
            per_screenshot={sid:dict(mean_seconds=float(np.mean(v)),repeat_seconds=v) for sid,v in by[method].items()})
    assert set(by['ours'])==set(by['mvp'])
    ratio=result['ours']['mean_seconds']/result['mvp']['mean_seconds']
    # Sampling uncertainty uses screenshots as independent units, not repetitions.
    ids=sorted(by['ours']);a=np.asarray([np.mean(by['ours'][s]) for s in ids]);b=np.asarray([np.mean(by['mvp'][s]) for s in ids])
    rng=np.random.default_rng(20261110);ix=rng.integers(0,len(ids),(5000,len(ids)))
    interval=np.quantile(a[ix].mean(1)/b[ix].mean(1),[.025,.975]).tolist()
    estimate={method:result[method]['mean_seconds']*1581/3600 for method in result}
    value=dict(complete=True,at_utc=now(),methods=result,ours_to_mvp_ratio=ratio,
        descriptive_screenshot_bootstrap_ratio_95=interval,
        full_1581_inference_hours_extrapolated=estimate,combined_hours=sum(estimate.values()),
        limitations='Six deliberately application-diverse timing screenshots; not representative full-benchmark latency, no accuracy comparison, fixed sequential method order, differing runtime hooks/device maps/prompts preserved. Extrapolation is not a deadline or completed benchmark. Fallback rate on6 is uncertain.')
    write(OUT/'analysis.json',value)
    lines=['V30 current-policy / MVP fresh single-query timing',
        'Same six ScreenSpot-Pro screenshots, two uncached repeats each; one separate warmup each.',
        'Model loading excluded from query time and reported separately; no accuracy measured.',
        'Ours is V29 keep_all_free, with output-equivalent conditional evidence scheduling, not V15 or the optional gate.', '']
    for method,m in result.items():
        lines += [f'{method}: mean={m["mean_seconds"]:.3f}s, median={m["median_seconds"]:.3f}s, range=[{m["min_seconds"]:.3f},{m["max_seconds"]:.3f}]s, model load={m["model_load_seconds"]:.2f}s',
            f'  generation calls={m["mean_generation_calls"]:.2f}, model forwards={m["mean_forward_calls"]:.2f}, generated tokens={m["mean_generated_tokens"]:.2f}, processed pixels={m["mean_processed_pixels"]:.0f}, fallback observations={m["fallback_queries"]}/12',
            '  stages: '+json.dumps(m['by_stage'],ensure_ascii=False)]
    lines += ['',f'Ours / MVP elapsed ratio={ratio:.4f}; descriptive screenshot bootstrap95%={interval}',
        '1581-query extrapolated GPU hours: '+json.dumps(estimate)+f'; combined={sum(estimate.values()):.2f}h',
        'Model loading, warmup, fallback distribution, runtime variation and final analysis add overhead.',
        'The earlier20-26h estimate concerned TWO complete1581-query runs, not a single prediction or training.',
        value['limitations']]
    (OUT/'readout.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    status('controller','complete',ratio=ratio,combined_estimate_hours=sum(estimate.values()))
    print('\n'.join(lines),flush=True)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','ours','mvp','analyze','all']);args=parser.parse_args()
    if args.stage=='prepare':prepare()
    elif args.stage in ['ours','mvp']:worker(args.stage)
    elif args.stage=='analyze':analyze()
    else:
        prepare()
        for method in ['ours','mvp']:
            status('controller','running_'+method)
            folder=OUT/method;folder.mkdir(parents=True,exist_ok=True)
            with (folder/'stdout.log').open('w',encoding='utf-8') as stdout,(folder/'stderr.log').open('w',encoding='utf-8') as stderr:
                process=subprocess.run([sys.executable,'-B','-X','utf8','-u',str(Path(__file__)),method],
                    cwd=ROOT,stdout=stdout,stderr=stderr)
            assert process.returncode==0, f'{method} failed; see {folder}/stderr.log'
        analyze()

if __name__=='__main__':
    try:main()
    except BaseException:
        status('controller','failed',traceback=traceback.format_exc());raise
