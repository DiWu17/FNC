"""Frozen3k/full x post-prediction visual feedback; local Python only."""
import os
os.environ.setdefault('PYTHONDONTWRITEBYTECODE', '1')
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src')]
import numpy as np
from PIL import Image, ImageDraw
from clickagent.visual_feedback_v23 import (ARMS, digest, make_pool, visual_inputs,
                                           decode_coordinate, decode_selection)

OUT = ROOT/'runs/visual_feedback_v23'
MAIN = ROOT/'runs/quality_readout_v15_3k'
FULL = ROOT/'runs/full_scaling_v18'
SOURCES = [Path(__file__), ROOT/'src/clickagent/visual_feedback_v23.py',
           ROOT/'tests/test_visual_feedback_v23.py']
METHODS = ['baseline']+list(ARMS)
FAMILIES = ['original3k', 'full']


def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def now(): return datetime.now(timezone.utc).isoformat()
def key(s): return hashlib.sha256(s.encode()).hexdigest()[:24]
def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(8*1024*1024), b''): h.update(b)
    return h.hexdigest()


def write(p, v):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(v, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    for i in range(40):
        try: os.replace(tmp, p); return
        except PermissionError:
            if i == 39: raise
            time.sleep(.1)


def status(phase, **fields):
    row = dict(phase=phase, complete=phase=='complete', updated_at_utc=now(), pid=os.getpid(), **fields)
    write(OUT/'status.json', row); print(json.dumps(row, ensure_ascii=False), flush=True)


def source_identity(): return {str(p):sha(p) for p in SOURCES}


def verify():
    p = read(OUT/'protocol.json')
    assert p['source_sha256'] == source_identity(), 'Source changed after registration'
    for name, h in p['sealed_sha256'].items(): assert sha(name)==h, name
    for name, row in p['backbone_identity'].items():
        st = Path(name).stat()
        assert st.st_size==row['bytes'] and st.st_mtime_ns==row['mtime_ns'], name
    return p


def prepare():
    if (OUT/'protocol.json').exists(): return verify()
    import torch
    from clickagent.evidence_retention import RetentionModel
    torch.set_num_threads(4)
    check = read(OUT/'unit_tests.json')
    assert check['exit_code']==0 and check['source_sha256']==source_identity()
    smoke_receipt = read(OUT/'smoke/receipt.json')
    assert smoke_receipt['complete'] and smoke_receipt['source_sha256']==source_identity()
    cases = read(ROOT/'runs/mvp_repair_v21/data/public.json')
    assert len(cases)==100 and len({c['group'] for c in cases})==100
    assert Counter(c['cohort'] for c in cases)==dict(new_development=72,old_val=28)
    indices = read(ROOT/'runs/bottleneck_v19/data/indices.json')
    entries = {e['sample_id']:e for co in ['old_val','new_development'] for e in indices[co]}
    train_groups = {e['group'] for e in indices['train']}
    with (FULL/'data/new_train.jsonl').open(encoding='utf-8-sig') as f:
        train_groups.update(json.loads(line)['pixel_sha256'] for line in f if line.strip())
    assert not train_groups & {c['group'] for c in cases}
    models = read(FULL/'models/frozen.json')['models']
    names = [f'{family}_s{s}' for family in FAMILIES for s in [1,2,3]]
    loaded = {}
    for name in names:
        info = models[name]; assert sha(info['path'])==info['sha256']
        saved = torch.load(info['path'], map_location='cpu', weights_only=True)
        model = RetentionModel().eval().requires_grad_(False); model.load_state_dict(saved['state_dict'])
        assert sum(p.numel() for p in model.parameters())==22050
        loaded[name] = (model, saved)
    labels = read(ROOT/'runs/mvp_repair_v21/data/labels.json')
    proposals = {}; provenance = {}; replay = []
    for j, c in enumerate(cases,1):
        assert sha(c['image'])==c['image_sha256']
        e = entries[c['sample_id']]
        for p,h in [('path','sha256'),('state_path','state_sha256'),('mask_path','mask_sha256')]:
            assert sha(e[p])==e[h]; provenance[e[p]]=e[h]
        with np.load(e['path'], allow_pickle=False) as z:
            points, base, active = [z[n].copy() for n in ['points','base','active']]
        state = np.load(e['state_path'], allow_pickle=False)
        mask = np.load(e['mask_path'], allow_pickle=False)
        assert len(points)==e['n']
        inputs = dict(state=torch.from_numpy(state), mask=torch.from_numpy(mask),active=torch.from_numpy(active))
        row = dict(points=points.tolist(), models={})
        with torch.inference_mode():
            for name,(model,saved) in loaded.items():
                inputs['base'] = torch.from_numpy((base-saved['mean'].numpy())/saved['std'].numpy())
                scores = model(**inputs).float().numpy() if len(points) else np.empty(0,np.float32)
                assert np.isfinite(scores).all()
                pool = make_pool(points,scores,c['image_size'])
                row['models'][name] = dict(scores=scores.tolist(), pool=pool, anchor=pool[0] if pool else None)
                if name.startswith('original3k'):
                    co='old_val' if c['cohort']=='old_val' else 'new_reserve'
                    path=MAIN/f'predictions/{co}/expanded_s{name[-1]}/{Path(e["path"]).stem}.npz'
                    with np.load(path,allow_pickle=False) as z: ref=z['scores'].copy()
                    assert ref.shape==scores.shape and np.allclose(ref,scores,atol=1e-4,rtol=1e-5), name
                    assert len(scores)==0 or int(ref.argmax())==int(scores.argmax())
                    provenance[str(path)] = sha(path)
                    replay.append(dict(sample_id=c['sample_id'],model=name,max_error=float(np.max(np.abs(ref-scores))) if len(scores) else 0.))
        proposals[c['sample_id']]=row
        if j%10==0: status('preparing_frozen_candidate_predictions',completed=j,total=len(cases))
    write(OUT/'data/public.json',cases);write(OUT/'data/labels.json',labels)
    write(OUT/'data/proposals.json',proposals);write(OUT/'data/provenance.json',provenance)
    write(OUT/'data/indices.json',{c['sample_id']:entries[c['sample_id']] for c in cases})
    write(OUT/'preflight.json',dict(original3k_score_replays=replay,train_rgb_overlap=0,models={n:models[n] for n in names}))
    # Only public hypotheses and image identity determine duplicate-call reuse.
    tasks = []
    for c in cases:
        pools={digest(v['pool']):v['pool'] for v in proposals[c['sample_id']]['models'].values() if v['pool']}
        for pool in sorted(pools.values(),key=digest):
            for arm in ARMS:
                tasks.append(dict(sample_id=c['sample_id'],pool=pool,arm=arm,
                                  task_id=digest([c['sample_id'],pool,arm])[:24]))
    write(OUT/'data/tasks.json',tasks)
    sealed = SOURCES+[OUT/f'data/{n}.json' for n in ['public','labels','proposals','provenance','indices','tasks']]
    sealed += [OUT/'unit_tests.json',OUT/'smoke/receipt.json',OUT/'preflight.json',FULL/'models/frozen.json',
               ROOT/'src/clickagent/evidence_retention.py',ROOT/'src/clickagent/advantage_runtimes.py',
               ROOT/'src/clickagent/public_evaluation.py',ROOT/'runs/mvp_repair_v21/protocol.json']
    sealed += [Path(models[n]['path']) for n in names]
    identity = read(ROOT/'data/conditioned_v3/qwen_weight_identity.json');backbone={}
    expected = read(ROOT/'runs/mvp_repair_v21/protocol.json')['backbone_files']
    for name,info in identity['files'].items():
        p=ROOT/'models/Qwen3-VL-8B-Instruct'/name;st=p.stat()
        old=expected[str(p)]
        assert st.st_size==info['bytes']==old['bytes'] and st.st_mtime_ns==old['mtime_ns'] and info['sha256']==old['sha256']
        backbone[str(p)]=dict(sha256=info['sha256'],bytes=st.st_size,mtime_ns=st.st_mtime_ns)
    spec=dict(registered_at_utc=now(),source_sha256=source_identity(),sealed_sha256={str(p):sha(p) for p in sealed},
        backbone_identity=backbone,model_names=names,models={n:models[n] for n in names},
        cohort='Same SHA-sampled V21 development100: new72(24/source),old28(7/source),one query/RGB. Historically reused; no fresh280 access. Exact training RGB overlap zero for original3k and full.',
        arms=dict(baseline='Frozen complete-pool argmax, each of6 checkpoints separately.',
            local_relook='Unmarked overview+initial-point crop, original coordinates and3 candidate hypotheses in text; free KEEP candidate / CLICK new full-image coordinates / WAIT.',
            marked_verify='Identical overview+crop pixels except red rings/labels; choose among exactly the same3 candidate points or N, constrained first-token score. No new coordinates.',
            marked_refine='Exactly local_relook prompt, images and generation rule, adding only the same pixel markers used by marked_verify; free KEEP / CLICK / WAIT.'),
        proposal_rule='Top1 anchor unchanged; add2 score-ranked distinct14px neighborhoods, falling back to distinct points; all content derived from predictions. Canonical extras, deterministic hash-permuted display shared by every arm. No candidate labels/geometry used.',
        observation_budget='All3 feedback arms exactly2 images, fixed512x512 each: full letterboxed overview plus original512-source-pixel crop around initial point. Same boxes/order/coordinates; no target-aware crops. Additional524288 pixels. One-round feedback only. Model loading excluded.',
        generation='Frozen Qwen3-VL-8B-Instruct FP16,2 GPUs,greedy,max32 generated tokens for coordinate arms,strict completed-EOS parser. Verify1 prefill constrains to letters A/B/C/N. Actual decode cost differs and is measured,not equal-compute assertion.',
        fallback='WAIT,N,malformed,truncated output retain original anchor in all arms; no learned confidence threshold or action rejection accuracy claim. Empty original pool remainsNone; denominator retained.',
        cache='Identical sample/pool/arm reused across6 fits; logical per-fit costs reported,not cache-read time. No seed ensemble. Actual calls are '+str(len(tasks))+'.',
        primary='Six contrasts: each feedback arm minus own original baseline for each training size,onnew72.5000 source-stratified RGB bootstrap over3-fit mean,99.1667% intervals (Bonferroni6).',
        secondary='Old28/all100,source/application grouped sensitivity,marker effect=marked_refine-local_relook (same prompt/observations),coordinate protocol effect=marked_refine-marked_verify (output budget differs),training-size interactions,repairs/harms,parse/fallback,new coordinate versus existing-pool hits.',
        advancement='Development signal only: >=3pp and primary adjusted lower>0,old28 delta>=-3pp. No automatic model replacement or fresh-test claim.',
        scope='Pure inference adaptation; no scorer/backbone training,real clicks,or asynchronous CoT injection. No HTML. Preserve all historical model/source/input files.')
    for p in SOURCES:
        dest=OUT/'source_archive'/p.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest)
    write(OUT/'registry_before.json',read(ROOT/'CURRENT_RESEARCH.json'));write(OUT/'protocol.json',spec)
    reg=read(ROOT/'CURRENT_RESEARCH.json');reg['active_experiment']=dict(run_id=OUT.name,status='registered',
        phase='prepared_frozen_models',protocol_path=str(OUT/'protocol.json'),status_path=str(OUT/'status.json'),
        mainline_unchanged=True,n=100,training=False,models=names,actual_feedback_calls=len(tasks))
    write(ROOT/'CURRENT_RESEARCH.json',reg)
    write(ROOT/'research/decisions/20261109_visual_feedback_v23.json',dict(registered_at_utc=now(),
        user_authorization='都用吧，分别比较局部重看、标记后复核、标记后允许修改坐标。',
        protocol_sha256=sha(OUT/'protocol.json'),mainline_unchanged=True))
    status('registered',screenshots=100,models=6,logical_feedback_calls=1800,actual_feedback_calls=len(tasks))
    return spec


def sync(torch):
    for d in range(torch.cuda.device_count()):torch.cuda.synchronize(d)


def load_runtime():
    from clickagent.advantage_runtimes import configure_paths
    configure_paths('coordinate')
    import torch
    from clickagent.advantage_runtimes import QwenCoordinateAdapter
    torch.set_num_threads(4);torch.manual_seed(20262301)
    runtime=QwenCoordinateAdapter(OUT/'runtime',max_pixels=262144)
    runtime.device=runtime.model.get_input_embeddings().weight.device
    return runtime,torch


def infer(runtime,torch,image,instruction,sample_id,pool,arm,preview=None):
    images,messages,geometry=visual_inputs(image,instruction,sample_id,pool,arm)
    if preview is not None:
        path=Path(preview);path.parent.mkdir(parents=True,exist_ok=True)
        canvas=Image.new('RGB',(1024,512),'white')
        for i,im in enumerate(images):canvas.paste(im,(512*i,0))
        canvas.save(path);canvas.close()
    sync(torch);before=runtime.counter.snapshot();started=time.perf_counter()
    text=runtime.processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
    inputs=runtime.processor(text=[text],images=images,return_tensors='pt',padding=False).to(runtime.device)
    grids=inputs['image_grid_thw'].detach().cpu().tolist();assert grids==[[1,32,32],[1,32,32]]
    generated_text=None;token_ids=[];selection_scores=None;mass=None;completed=None
    with torch.inference_mode():
        if arm=='marked_verify':
            output=runtime.model(**inputs,use_cache=False,logits_to_keep=1,return_dict=True)
            logits=output.logits[0,-1].float().cpu();del output
            groups=[]
            for label in geometry['labels']+['N']:
                group=set()
                for prefix in ['',' ']:
                    ids=runtime.processor.tokenizer.encode(prefix+label,add_special_tokens=False)
                    if len(ids)==1:group.add(ids[0])
                assert group;groups.append(sorted(group))
            assert len({i for group in groups for i in group})==sum(map(len,groups))
            scores=torch.stack([torch.logsumexp(logits[g],0) for g in groups])
            selection_scores=scores.tolist();mass=float((torch.logsumexp(scores,0)-torch.logsumexp(logits,0)).exp())
            result=decode_selection(selection_scores,geometry)
        else:
            length=inputs['input_ids'].shape[-1]
            output=runtime.model.generate(**inputs,max_new_tokens=32,do_sample=False,use_cache=True,
                                          return_dict_in_generate=False)
            token_ids=output[0,length:].detach().cpu().tolist();del output
            generated_text=runtime.processor.tokenizer.decode(token_ids,skip_special_tokens=True).strip()
            eos=runtime.model.generation_config.eos_token_id;eos=eos if isinstance(eos,list) else [eos]
            completed=bool(token_ids and token_ids[-1] in eos)
            result=decode_coordinate(generated_text,completed,geometry)
    sync(torch);after=runtime.counter.snapshot()
    result.update(geometry=geometry,generated_text=generated_text,generated_token_ids=token_ids,
        terminated_by_eos=completed,selection_scores=selection_scores,valid_label_mass=mass,
        input_tokens=int(inputs['input_ids'].shape[-1]),image_grids=grids,
        elapsed_seconds=time.perf_counter()-started,counter={k:after[k]-before[k] for k in before},
        ground_truth_in_model_inputs=False)
    assert result['counter']['processed_pixels']==524288
    assert result['counter']['backbone_forward_calls']>=1
    for im in images:im.close()
    return result


def smoke():
    OUT.mkdir(parents=True,exist_ok=True);status('synthetic_interface_check_loading')
    runtime,torch=load_runtime();rows=[]
    for case in [0,1]:
        image=Image.new('RGB',(1200,800),'white');draw=ImageDraw.Draw(image)
        draw.rectangle((100,250,350,450),fill=(220,20,20));draw.rectangle((750,250,1000,450),fill=(20,70,220))
        pool=[[220,350],[870,350],[580,650]] if case==0 else [[870,350],[220,350],[580,650]]
        for arm in ARMS:
            result=infer(runtime,torch,image,'Click the blue rectangle on the right.',f'synthetic{case}',pool,arm,
                         OUT/f'smoke/{case}_{arm}.png')
            assert result['valid'], (arm,result['generated_text'])
            result.update(case=case,arm=arm,semantic_success=750<=result['point'][0]<=1000 and 250<=result['point'][1]<=450)
            rows.append(result);status('synthetic_interface_check',case=case,arm=arm,kind=result['kind'],point=result['point'])
        image.close()
    write(OUT/'smoke/receipt.json',dict(complete=True,at_utc=now(),source_sha256=source_identity(),
        synthetic_only=True,observations=rows,runtime=runtime.metadata,
        semantic_success=sum(r['semantic_success'] for r in rows),semantic_not_a_gate=True))
    status('synthetic_interface_check_complete',calls=len(rows))


def worker():
    verify();cases={c['sample_id']:c for c in read(OUT/'data/public.json')};tasks=read(OUT/'data/tasks.json')
    protocol_sha=sha(OUT/'protocol.json');runtime=torch=None;started=time.perf_counter();new_calls=0
    for i,task in enumerate(tasks,1):
        path=OUT/f'calls/{task["task_id"]}.json'
        if path.exists():
            saved=read(path);assert saved['protocol_sha256']==protocol_sha and saved['task']==task
            continue
        if runtime is None:
            status('loading_frozen_feedback_model',completed=i-1,total=len(tasks));runtime,torch=load_runtime()
            write(OUT/'runtime/metadata.json',runtime.metadata)
        c=cases[task['sample_id']];assert sha(c['image'])==c['image_sha256']
        status('feedback_inference',completed=i-1,total=len(tasks),arm=task['arm'],sample_id=c['sample_id'],
               new_calls=new_calls,elapsed_seconds=time.perf_counter()-started)
        with Image.open(c['image']) as opened:image=opened.convert('RGB')
        result=infer(runtime,torch,image,c['instruction'],c['sample_id'],task['pool'],task['arm'],
                     OUT/f'previews/{task["task_id"]}.png' if i<=6 else None)
        image.close();new_calls+=1
        write(path,dict(task=task,output=result,protocol_sha256=protocol_sha,image_sha256=c['image_sha256']))
    status('feedback_inference_complete',completed=len(tasks),total=len(tasks),new_calls=new_calls)


def hit(point,case,geometry):
    from clickagent.public_evaluation import point_in_geometry
    return bool(point is not None and point_in_geometry([p/s for p,s in zip(point,case['image_size'])],geometry))


def bootstrap(diff,cases,level,group_field='group'):
    strata=defaultdict(lambda:defaultdict(list))
    for i,c in enumerate(cases):strata[c['source']][c[group_field]].append(i)
    groups=[list(v.values()) for v in strata.values()];rng=np.random.default_rng(20262302)
    v=np.asarray(diff).mean(0);draws=[]
    for _ in range(5000):
        ix=[i for g in groups for j in rng.integers(0,len(g),len(g)) for i in g[j]]
        draws.append(float(v[ix].mean()))
    a=(1-level)/2
    return dict(delta_pp=100*float(v.mean()),ci_pp=(100*np.quantile(draws,[a,1-a])).tolist(),
                interval_percent=100*level,seed_delta_pp=(100*np.asarray(diff).mean(1)).tolist())


def analyze():
    spec=verify();cases=read(OUT/'data/public.json');labels=read(OUT/'data/labels.json')
    proposals=read(OUT/'data/proposals.json');tasks=read(OUT/'data/tasks.json');protocol_sha=sha(OUT/'protocol.json')
    calls={};actual=[]
    for task in tasks:
        p=OUT/f'calls/{task["task_id"]}.json';row=read(p)
        assert row['task']==task and row['protocol_sha256']==protocol_sha
        o=row['output'];g=o['geometry']
        expected=decode_selection(o['selection_scores'],g) if task['arm']=='marked_verify' else decode_coordinate(o['generated_text'],o['terminated_by_eos'],g)
        assert all(o[k]==v for k,v in expected.items()),p
        assert o['counter']['processed_pixels']==524288 and not o['ground_truth_in_model_inputs']
        calls[task['task_id']]=o;actual.append(dict(task_id=task['task_id'],sha256=sha(p)))
    predictions=[];vectors={};outputs={};replay_count=0
    for family in FAMILIES:
        for s in [1,2,3]:
            name=f'{family}_s{s}';vectors[name]={a:[] for a in METHODS};outputs[name]={a:[] for a in METHODS}
            for c in cases:
                sid=c['sample_id'];p=proposals[sid]['models'][name];base=p['anchor']
                all_points=proposals[sid]['points'];oracle=any(hit(q,c,labels[sid]) for q in all_points)
                pool_oracle=any(hit(q,c,labels[sid]) for q in p['pool'])
                for arm in METHODS:
                    task_id=None if arm=='baseline' or not p['pool'] else digest([sid,p['pool'],arm])[:24]
                    o=calls[task_id] if task_id else dict(point=base,kind='baseline' if arm=='baseline' else 'empty_pool',fallback=False,valid=True,elapsed_seconds=0,counter={})
                    correct=hit(o['point'],c,labels[sid]);vectors[name][arm].append(int(correct));outputs[name][arm].append(o)
                    predictions.append(dict(sample_id=sid,model=name,arm=arm,point=o['point'],hit=correct,baseline_point=base,
                        original_pool_oracle=oracle,shortlist_oracle=pool_oracle,task_id=task_id,kind=o['kind'],
                        point_in_original_pool=o['point'] in all_points if o['point'] is not None else False))
                    replay_count+=1
    # Baseline parity with sealed historical evaluation, checked only after every feedback prediction exists.
    parity=0
    for co in ['old_val','new_development']:
        old=read(FULL/f'evaluation/{co}.json');lookup={sid:i for i,sid in enumerate(old['sample_ids'])}
        for name in vectors:
            for i,c in enumerate(cases):
                if c['cohort']==co:
                    assert bool(vectors[name]['baseline'][i])==old['hit_vectors'][name][lookup[c['sample_id']]],(name,c['sample_id'])
                    parity+=1
    result={};lines=['V23：3k/全量 × 局部重看、标记后复核、标记后允许修改坐标。',
        '100张历史复用开发图：新72/旧28，每图一查询。每版3固定种子，报告均值而非集成。',
        '完整候选初始点不变；所有反馈仅一轮，均为512概览+512原图局部。',
        '局部重看/标记后修正提示词相同，仅标记像素不同。复核只能选原三点，修正可输出新坐标。',
        'WAIT/N/无效输出保留原点。复核是单prefill约束字母分数，坐标组有解码，实际计算不同，单列耗时。']
    for co in ['new_development','old_val','all']:
        ix=[i for i,c in enumerate(cases) if co=='all' or c['cohort']==co];selected=[cases[i] for i in ix]
        arrays={f:{a:np.asarray([vectors[f'{f}_s{s}'][a] for s in [1,2,3]],float)[:,ix] for a in METHODS} for f in FAMILIES}
        summaries={};comparisons={};lines+=['',f'{co}, n={len(ix)}']
        for f in FAMILIES:
            summaries[f]={}
            for arm,v in arrays[f].items():
                outs=[outputs[f'{f}_s{s}'][arm][i] for s in [1,2,3] for i in ix]
                b=arrays[f]['baseline'];summary=dict(accuracy_percent=100*float(v.mean()),seed_hits=v.sum(1).astype(int).tolist(),
                    repaired_seed_counts=((v==1)&(b==0)).sum(1).astype(int).tolist(),
                    harmed_seed_counts=((v==0)&(b==1)).sum(1).astype(int).tolist(),
                    mean_extra_seconds=float(np.mean([o['elapsed_seconds'] for o in outs])),
                    median_extra_seconds=float(np.median([o['elapsed_seconds'] for o in outs])),
                    kind_counts=dict(Counter(o['kind'] for o in outs)),fallback_count=sum(o['fallback'] for o in outs),
                    invalid_count=sum(not o['valid'] for o in outs),
                    by_source={source:100*float(v[:,[j for j,c in enumerate(selected) if c['source']==source]].mean()) for source in sorted({c['source'] for c in selected})})
                summaries[f][arm]=summary
                lines.append(f'{f}/{arm}: {summary["accuracy_percent"]:.2f}%, 命中{summary["seed_hits"]},修复{summary["repaired_seed_counts"]},误改{summary["harmed_seed_counts"]},额外均时{summary["mean_extra_seconds"]:.2f}s')
                if arm!='baseline':
                    comp=bootstrap(v-b,selected,1-.05/6 if co=='new_development' else .95)
                    comparisons[f'{f}/{arm}-baseline']=comp
                    lines.append(f'  净增{comp["delta_pp"]:+.2f}pp; {comp["interval_percent"]:.3f}%区间{comp["ci_pp"]}')
            for label,a,b in [('marker_effect','marked_refine','local_relook'),('coordinate_protocol','marked_refine','marked_verify')]:
                comparisons[f'{f}/{label}']=bootstrap(arrays[f][a]-arrays[f][b],selected,.95)
        for arm in METHODS:
            comparisons[f'full-original3k/{arm}']=bootstrap(arrays['full'][arm]-arrays['original3k'][arm],selected,.95)
            if arm!='baseline':
                comparisons[f'training_size_feedback_interaction/{arm}']=bootstrap(
                    arrays['full'][arm]-arrays['full']['baseline']-arrays['original3k'][arm]+arrays['original3k']['baseline'],selected,.95)
        result[co]=dict(n=len(ix),methods=summaries,comparisons=comparisons)
        if co=='new_development':
            result[co]['application_group_sensitivity95']={f'{f}/{arm}':bootstrap(arrays[f][arm]-arrays[f]['baseline'],selected,.95,'source_group') for f in FAMILIES for arm in ARMS}
    gates={}
    for f in FAMILIES:
        for a in ARMS:
            name=f'{f}/{a}-baseline';v=result['new_development']['comparisons'][name];old=result['old_val']['comparisons'][name]
            gates[name]=v['delta_pp']>=3 and v['ci_pp'][0]>0 and old['delta_pp']>=-3
    lines+=['','开发正向门槛：'+str(gates),
        '以上不是新的独立测试，也不是实际动作执行反馈。未训练模型、未采用新主线、未读原fresh280。',
        f'全部{len(tasks)}次独立反馈调用完成，{replay_count}条最终预测重放、{parity}条基线历史命中向量一致。']
    write(OUT/'predictions.json',dict(rows=predictions,hit_vectors=vectors))
    write(OUT/'results.json',dict(complete=True,cohorts=result,positive_gates=gates,independent_confirmation=False,mainline_unchanged=True))
    (OUT/'results.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    write(OUT/'verification.json',dict(complete=True,call_count=len(tasks),prediction_replays=replay_count,
        baseline_hit_vector_parity=parity,all_calls_sha256=actual,protocol_sha256=protocol_sha,source_sha256=source_identity()))
    plot(result)
    verify()
    paths=[OUT/p for p in ['protocol.json','results.json','results.txt','predictions.json','verification.json','summary_zh.png']]
    write(OUT/'completion.json',dict(complete=True,completed_at_utc=now(),artifact_sha256={str(p):sha(p) for p in paths},mainline_unchanged=True))
    reg=read(ROOT/'CURRENT_RESEARCH.json');before=read(OUT/'registry_before.json')
    assert reg['models']==before['models'] and reg['run_id']==before['run_id']
    reg['active_experiment'].update(status='complete',phase='inference_evaluation_replay_analysis_complete',
        results_path=str(OUT/'results.txt'),completion_path=str(OUT/'completion.json'),positive_gates=gates)
    reg['completed_experiments'].append(dict(run_id=OUT.name,status='complete',analysis_path=str(OUT/'results.txt'),mainline_unchanged=True))
    write(ROOT/'CURRENT_RESEARCH.json',reg)
    write(ROOT/'research/decisions/20261109_visual_feedback_v23_closeout.json',dict(completed_at_utc=now(),
        positive_gates=gates,mainline_unchanged=True,readout=str(OUT/'results.txt'),independent_confirmation=False))
    status('complete',actual_feedback_calls=len(tasks),prediction_replays=replay_count,positive_gates=gates)
    print('\n'.join(lines),flush=True)


def plot(result):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif']=['Microsoft YaHei','SimHei','DejaVu Sans'];plt.rcParams['axes.unicode_minus']=False
    fig,axes=plt.subplots(1,2,figsize=(12,4.6));x=np.arange(4)
    for ax,co,title in zip(axes,['new_development','old_val'],['新开发72张','旧开发28张']):
        for f,offset,color in [('original3k',-.18,'#2878B5'),('full',.18,'#E78A32')]:
            vals=[result[co]['methods'][f][a]['accuracy_percent'] for a in METHODS]
            bars=ax.bar(x+offset,vals,.34,label='3k' if f=='original3k' else '全量',color=color)
            ax.bar_label(bars,fmt='%.1f',fontsize=9,padding=3)
        ax.set_xticks(x,['原流程','局部重看','标记后复核','标记后修正']);ax.set_ylim(0,100)
        ax.set_title(title);ax.set_ylabel('准确率（%）');ax.legend()
    fig.suptitle('V23：预测后的视觉反馈（3个固定种子均值，非集成）')
    fig.tight_layout();fig.savefig(OUT/'summary_zh.png',dpi=170);plt.close(fig)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['smoke','prepare','worker','analyze','all'],default='all');args=parser.parse_args()
    try:
        if args.stage=='smoke':smoke();return
        if args.stage in ['prepare','all']:prepare()
        if args.stage in ['worker','all']:worker()
        if args.stage in ['analyze','all']:analyze()
    except Exception as exc:
        status('failed',error=str(exc),traceback=traceback.format_exc());raise


if __name__=='__main__':main()
