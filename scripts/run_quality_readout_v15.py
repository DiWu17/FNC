"""V15 data scaling: immutable legacy inputs, resumable extraction/train/evaluation.

The candidate universe and 22,050-parameter readout are unchanged. No benchmark
test predictions, GUI teachers, labels or target boxes enter feature extraction.
"""
from __future__ import annotations

import os
os.environ.setdefault('PYTHONDONTWRITEBYTECODE', '1')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import gc
import hashlib
import html
import json
import math
from pathlib import Path
import shutil
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
sys.path.insert(0, str(ROOT/'scripts'))
import numpy as np
from PIL import Image
import torch
from clickagent.crossfit_proposer import load_crossfit_proposer, predict_raw
from clickagent.evidence_retention import (RetentionModel, STATE_DIM, build_base,
    state_features, retention_loss, require)
from clickagent.model_features import ExtractionConfig
from clickagent.multiregion_proposals import build_c1_c3_pools, plan_multiregion_crops
from clickagent.public_evaluation import point_in_geometry
from run_evidence_retention_v14 import tensors, paired

OUT = ROOT/'runs/quality_readout_v15'
OLD = ROOT/'runs/evidence_retention_v14'
DATA = ROOT/'data/expansion_v15/manifests'
V13 = ROOT/'runs/multiregion_v13'
V12 = ROOT/'runs/crossfit_candidates_v12'
SEEDS = (20261311, 20261312)
STEPS = 13400  # 8 * ceil(13397 / 8), fixed before any new feature/model outcome.
PUBLIC = ('sample_id', 'dataset', 'image', 'image_size', 'image_sha256', 'instruction')
SOURCES = (Path(__file__), ROOT/'src/clickagent/evidence_retention.py',
    ROOT/'scripts/run_evidence_retention_v14.py', ROOT/'src/clickagent/crossfit_proposer.py',
    ROOT/'src/clickagent/multiregion_proposals.py', ROOT/'src/clickagent/conditioned_features.py',
    ROOT/'src/clickagent/model_features.py', ROOT/'src/clickagent/target_token.py',
    ROOT/'src/clickagent/action_model.py', ROOT/'src/clickagent/public_evaluation.py',
    ROOT/'src/clickagent/public_model.py', ROOT/'src/clickagent/candidate_ranker.py',
    ROOT/'src/clickagent/semantic_head_ranker.py', ROOT/'src/clickagent/attention_field_ranker.py',
    ROOT/'src/clickagent/point_aligned_readout.py', ROOT/'src/clickagent/point_aligned_inference.py',
    ROOT/'tests/test_quality_readout_v15.py')


def now(): return datetime.now(timezone.utc).isoformat()
def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def rows(p): return [json.loads(l) for l in Path(p).read_text(encoding='utf-8-sig').splitlines() if l.strip()]
def key(s): return hashlib.sha256(s.encode()).hexdigest()[:24]
def digest(x): return hashlib.sha256(json.dumps(x, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''): h.update(block)
    return h.hexdigest()


def replace(temp, path):
    # A report reader can momentarily hold a Windows handle across replacement.
    for attempt in range(40):
        try: Path(temp).replace(path); return
        except PermissionError:
            if attempt == 39: raise
            time.sleep(.1)


def write(p, value):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    temp = p.with_suffix(p.suffix+'.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    replace(temp, p)


def npz(p, **arrays):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    temp = p.with_suffix(p.suffix+'.tmp')
    with temp.open('wb') as f: np.savez_compressed(f, **arrays)
    replace(temp, p)


def identity(): return {str(p.relative_to(ROOT)): sha(p) for p in SOURCES}


def legacy_hashes():
    found = {}
    c = read(V13/'completion.json')
    for base, section in ((V13, 'artifact_sha256'), (ROOT, 'docs_sha256')):
        for name, expected in c[section].items():
            p = base/name; require(sha(p) == expected, 'V13 completion drift: '+str(p)); found[str(p)] = expected
    c = read(OLD/'report_completion.json')
    # V14 report manifest has absolute artifact paths.
    for name, expected in c['files'].items():
        p = Path(name) if Path(name).is_absolute() else OLD/name
        require(sha(p) == expected, 'V14 completion drift: '+str(p)); found[str(p)] = expected
    for p in (V13/'frozen.json', OLD/'base/index.json', OLD/'states/index.json'):
        found[str(p)] = sha(p)
    return found


def render():
    state = read(OUT/'status.json') if (OUT/'status.json').exists() else {}
    body = ['<h1>V15：数据扩容后的质量评分器训练与测试</h1>',
        '<p>固定 Qwen、提议器、裁剪、候选坐标和 22,050 参数评分器；仅改变训练数据与对应训练对照。</p>',
        '<p>训练：原 873 + 新 12,524 = 13,397 题。评估：旧验证 172 题；新开发预留 1,439 题。</p>',
        '<p>新预留按软件/网站/移动 episode 分组，不宣称全局未见应用；旧验证有历史复用。本轮没有新的论文最终独立测试。</p>',
        '<p><a href="protocol.json">冻结实验协议</a> · <a href="status.json">实时进度</a> · <a href="logs/pipeline.log">运行日志</a></p>',
        '<pre>'+html.escape(json.dumps(state, ensure_ascii=False, indent=2))+'</pre>']
    for split in ('old_val', 'new_reserve'):
        p = OUT/f'evaluation/{split}/results.json'
        if not p.exists(): continue
        result = read(p)
        body.append('<h2>'+('旧验证集' if split == 'old_val' else '新开发预留集')+'</h2>')
        body.append('<table><tr><th>模型</th><th>Top1 命中 / 总题数</th><th>Top4 可达</th></tr>')
        for name, v in result['methods'].items():
            body.append(f'<tr><td>{html.escape(name)}</td><td>{v["old_k16_top1_hits"]} / {result["n"]}</td><td>{v["old_k16_top4_coverage"]}</td></tr>')
        body.append('</table><p>Top1 是强制选点的几何命中；Top4 可达不是 GUI 最终命中。所有模型读取逐题相同的原 K16 候选。</p>')
    content = '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta http-equiv="refresh" content="60"><title>ClickAgent V15 training</title><style>body{max-width:1100px;margin:36px auto;padding:0 24px;font:16px/1.7 Segoe UI,Microsoft YaHei,sans-serif;background:#f8fafc;color:#23354b}pre{white-space:pre-wrap;background:white;padding:18px}table{width:100%;border-collapse:collapse}th,td{text-align:left;padding:8px;border-bottom:1px solid #ddd}a{color:#176e9e}</style><body>'+'\n'.join(body)+'</body></html>'
    p = OUT/'report.html'; temp = p.with_suffix('.html.tmp'); temp.write_text(content, encoding='utf-8'); replace(temp, p)


def status(phase, **extra):
    value = dict(complete=False, phase=phase, updated_at_utc=now(), pid=os.getpid(), **extra)
    write(OUT/'status.json', value); render()
    print(json.dumps(value, ensure_ascii=False), flush=True)


def register():
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT/'protocol.json'
    if p.exists():
        protocol = read(p); require(protocol['source_identity'] == identity(), 'V15 source drift after registration')
        for name, expected in protocol['input_sha256'].items(): require(sha(name) == expected, 'Input drift: '+name)
        return protocol
    complete = read(ROOT/'runs/dataset_expansion_v15/completion.json'); require(complete['complete'], 'Import/QA incomplete')
    for name, expected in complete['artifact_sha256'].items():
        require(sha(ROOT/name)==expected, 'Imported artifact drift: '+name)
    manifests = {str(DATA/name): sha(DATA/name) for name in ('train.jsonl', 'validation_reserve.jsonl', 'combined_train.jsonl')}
    counts = {name: len(rows(DATA/name)) for name in ('train.jsonl', 'validation_reserve.jsonl', 'combined_train.jsonl')}
    require(counts == {'train.jsonl':12524, 'validation_reserve.jsonl':1439, 'combined_train.jsonl':13397}, 'Wrong cohort')
    inputs = {**manifests, **legacy_hashes()}
    for p0 in (ROOT/'runs/dataset_expansion_v15/completion.json', V12/'runtime_identity.json', V12/'proposers/full.pt'):
        inputs[str(p0)] = sha(p0)
    for variant in ('rank', 'aligned', 'permuted'):
        for s in (1, 2):
            for suffix in ('.pt', '.json'):
                p0 = OLD/f'models/{variant}_s{s}{suffix}'; inputs[str(p0)] = sha(p0)
    spec = dict(version=15, created_at_utc=now(), source_identity=identity(), input_sha256=inputs,
        question='数据扩容是否在固定候选池与固定模型下改善质量评分、跨来源表现与两种子稳定性？',
        counts=counts, seeds=list(SEEDS), parameters=22050,
        upstream='Unchanged frozen Qwen8B float16, layers15/28/36, max_pixels262144, natural own target-name generation; sealed V12 full proposer on all new samples. Legacy train retains authenticated OOF upstream. No retraining of Qwen/proposer/GUI verifier.',
        feature_inputs='Public screenshot/instruction only, max3 unchanged label-free local crops; all decoded coordinates and original K16. Exact V14 32 base +294 state channels; all target-name tokens,96 heads.',
        extraction='Raw six-array archives retained and SHA-bound; feature builders accept no GT. Save compact state arrays per query; deterministic sample-ID SHA order within each new cohort. No native-coordinate run required for this data-scaling comparison.',
        arms={'old8':'All6 archived V14 rank/aligned/permuted S1/S2 epoch8 references, no retraining',
            'old_matched':'Aligned S1/S2 from identical random initialization on old873, same13400 optimizer updates as expanded; separates additional training compute from expanded data',
            'expanded':'Rank/aligned/permuted S1/S2 from identical random initialization on all13397 combined queries, fixed13400 optimizer updates (8 nominal passes); zero-candidate records skipped in fitting but retained in all evaluation denominators'},
        optimization=dict(steps=STEPS,batch_queries=8,lr=.0002,weight_decay=.02,gradient_clip=1.,dropout=0,
            loss='Unchanged multipositive listwise/null +.25 balanced BCE over complete decoded union; per-training-regime base normalization fitted on train only',
            schedule='Independent deterministic permutation each epoch; complete or final partial query batches. Stop exactly at13400 updates. Save/resume optimizer+RNG+position, no validation-selected checkpoint.'),
        primary='Forced quality argmax among unchanged original actual K16, ties in canonical union-index order. S1 primary, S2 robustness; do not choose winning seed.',
        secondary=['Top4 coverage within original K16','Fixed original K16 coverage and whole-union coverage ceilings', 'Direct whole-union Top1 separately, since more points are scored'],
        evaluation='Freeze all8 new models before reserve extraction/prediction. Evaluate all14 checkpoints on same old_val172 and new_reserve1439; no new-policy selection using old192 test or reserve outcomes. No GUI verifier training/scoring in this isolated first data-scaling experiment.',
        comparisons=['expanded aligned vs old8 aligned, eachseed','expanded aligned vs old_matched aligned, eachseed','expanded aligned vs expanded rank/permuted, eachseed'],
        uncertainty='Source-stratified RGB-cluster paired bootstrap2000, plus source-group cluster interval on new reserve. Conditional on fixed trained models; source-group counts reported.',
        limitations=['Old validation historically reused; new reserve is development, not official independent benchmark.',
            'New-data groups disjoint within new import only; not globally unseen apps, mobile episode proxies are not app labels.',
            'New samples use full legacy proposer; old training uses OOF proposers. This pre-existing upstream asymmetry is fixed across readout comparisons, not a new end-to-end crossfit claim.',
            'No click execution or end-to-end task-success measurement. Top4 coverage is not verifier success.',
            'Frozen backbone pretraining overlap unknown; exact/near screenshot filtering cannot prove semantic independence.'])
    archive = OUT/'source_archive'; archive.mkdir(exist_ok=True)
    for source in SOURCES:
        destination = archive/source.relative_to(ROOT); destination.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(source,destination)
    write(p, spec); status('protocol_registered', optimizer_steps=STEPS, counts=counts)
    return spec


def verify():
    p = read(OUT/'protocol.json'); require(p['source_identity'] == identity(), 'V15 source changed')
    for name, expected in p['input_sha256'].items(): require(sha(name) == expected, 'Frozen input changed: '+name)


def old_entries(split):
    base = read(OLD/'base/index.json')['cases']; states = {e['sample_id']:e for e in read(OLD/'states/index.json')['cases']}
    cases = {c['sample_id']:c for c in rows(V13/f'data/{split}.jsonl')}
    result = []
    for b in base:
        if b['split'] != split: continue
        s = states[b['sample_id']]; c = cases[b['sample_id']]
        result.append(dict(b, state_path=s['path'], mask_path=s['mask_path'], state_sha256=s['sha256'], mask_sha256=s['mask_sha256'],
            source_group=c['pixel_sha256'], role='old_train' if split=='train' else 'old_val'))
    return result


def load_records(entries):
    result = []
    for e in entries:
        for p, h in (('path','sha256'),('labels_path','labels_sha256'),('state_path','state_sha256'),('mask_path','mask_sha256')):
            require(sha(e[p]) == e[h], 'Feature identity drift: '+e[p])
        with np.load(e['path'],allow_pickle=False) as z: r = {k:z[k].copy() for k in ('points','base','active','baseline')}
        r.update(e); r['labels'] = np.load(e['labels_path'],allow_pickle=False)
        r['state'] = np.load(e['state_path'],mmap_mode='r',allow_pickle=False)
        r['mask'] = np.load(e['mask_path'],mmap_mode='r',allow_pickle=False)
        result.append(r)
    return result


def model_name(regime, variant, seed): return f'{regime}_{variant}_s{SEEDS.index(seed)+1}'


def next_batch(n,seed,epoch,position):
    require(n>0 and 0<=position<=n,'Invalid training cursor')
    if position==n:epoch+=1;position=0
    order=np.random.default_rng(seed+epoch*101).permutation(n)
    indices=order[position:position+8]
    return indices,epoch,position+len(indices)


def train_one(regime, variant, seed, device):
    name = model_name(regime,variant,seed); receipt = OUT/f'models/{name}.json'
    if receipt.exists():
        r=read(receipt); require(r['complete'] and sha(r['checkpoint_path'])==r['checkpoint_sha256'],'Invalid finished model'); return
    entries = old_entries('train')
    if regime == 'expanded':
        ix = read(OUT/'features/new_train/index.json'); require(ix['complete'],'New train features incomplete'); entries += ix['cases']
    records = [r for r in load_records(entries) if r['n']]
    require(records, 'No fitting records')
    x = np.concatenate([r['base'] for r in records]); mean=x.mean(0); std=x.std(0); std[std<1e-5]=1; del x
    torch.manual_seed(seed); np.random.seed(seed)
    model=RetentionModel().to(device)
    initial=digest({k:hashlib.sha256(v.detach().cpu().numpy().tobytes()).hexdigest() for k,v in model.state_dict().items()})
    reference=read(OLD/f'models/{variant}_s{SEEDS.index(seed)+1}.json')
    require(initial==reference['initial_state_sha256'],'Initialization differs from V14')
    optimizer=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=.02)
    history=[]; step=0; epoch=1; position=0; elapsed=0.; resumed=False
    resume=OUT/f'models/{name}.resume.pt'
    if resume.exists():
        saved=torch.load(resume,map_location='cpu',weights_only=True)
        require(saved['protocol_sha256']==sha(OUT/'protocol.json'),'Resume protocol drift')
        require(saved['entry_identity']==digest(entries),'Resume input order drift')
        model.load_state_dict(saved['state_dict']); optimizer.load_state_dict(saved['optimizer'])
        step=saved['step']; epoch=saved['epoch']; position=saved['position']; history=saved['history']; elapsed=saved['elapsed_seconds']
        torch.set_rng_state(saved['torch_rng'])
        if device.startswith('cuda'): torch.cuda.set_rng_state_all(saved['cuda_rng'])
        resumed=True
    started=time.perf_counter(); model.train()
    status('training',model=name,step=step,total_steps=STEPS,fitting_queries=len(records),excluded_empty=len(entries)-len(records),device=device,resumed=resumed)
    losses=[]
    while step<STEPS:
        indices,epoch,position=next_batch(len(records),seed,epoch,position)
        batch=[records[i] for i in indices]
        inputs,lengths=tensors(batch,mean,std,variant,device)
        optimizer.zero_grad(set_to_none=True); scores=model(**inputs)
        loss=torch.stack([retention_loss(s,torch.as_tensor(r['labels'],device=device),model.null) for s,r in zip(scores.split(lengths),batch)]).mean()
        require(bool(torch.isfinite(loss)),'Nonfinite train loss')
        loss.backward();require(all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters()),'Nonfinite gradient')
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
        step+=1;losses.append(float(loss.detach()))
        if step%250==0 or step==STEPS:
            h=dict(step=step,epoch=epoch,position=position,train_loss=float(np.mean(losses)),elapsed_seconds=elapsed+time.perf_counter()-started)
            history.append(h); losses=[]
            temp=resume.with_suffix('.pt.tmp');temp.parent.mkdir(parents=True,exist_ok=True)
            torch.save(dict(state_dict=model.state_dict(),optimizer=optimizer.state_dict(),step=step,epoch=epoch,position=position,
                history=history,elapsed_seconds=h['elapsed_seconds'],protocol_sha256=sha(OUT/'protocol.json'),entry_identity=digest(entries),
                torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if device.startswith('cuda') else []),temp)
            replace(temp,resume)
            write(OUT/f'models/{name}_progress.json',dict(name=name,total_steps=STEPS,**h))
            status('training',model=name,total_steps=STEPS,fitting_queries=len(records),**h)
    model.eval();ckpt=OUT/f'models/{name}.pt';temp=ckpt.with_suffix('.pt.tmp')
    torch.save(dict(state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},mean=torch.tensor(mean),std=torch.tensor(std),variant=variant,seed=seed),temp);replace(temp,ckpt)
    write(receipt,dict(complete=True,name=name,regime=regime,variant=variant,seed=seed,steps=STEPS,checkpoint_path=str(ckpt),
        checkpoint_sha256=sha(ckpt),initial_state_sha256=initial,parameter_count=sum(p.numel() for p in model.parameters()),
        manifest_queries=len(entries),fitting_queries=len(records),excluded_empty=len(entries)-len(records),entry_identity=digest(entries),
        protocol_sha256=sha(OUT/'protocol.json'),history=history,normalization_training_only=True,validation_used_for_selection=False))
    del records,model,optimizer;gc.collect()
    if torch.cuda.is_available():torch.cuda.empty_cache()


class Extractor:
    def __init__(self):
        record=read(V12/'runtime_identity.json');contract=record['runtime_contract']
        for e in record['actual_model_files']:
            s=Path(e['path']).stat();require(s.st_size==e['bytes'] and s.st_mtime_ns==e['mtime_ns'],'Qwen weight inventory drift')
        self.contract=contract;cfg=dict(contract);cfg['layers']=tuple(cfg['layers']);cfg['conditions']=tuple(cfg['conditions'])
        cfg['max_memory']={int(k) if str(k).isdigit() else k:v for k,v in cfg['max_memory'].items()}
        cfg['offload_folder']=str(OUT/'offload');self.config=ExtractionConfig(**cfg)
        cfg2=asdict(self.config);cfg2.pop('offload_folder');require(json.loads(json.dumps(cfg2))==contract,'Qwen contract drift')
        self.proposer=load_crossfit_proposer(V12/'proposers/full.pt','cpu')[0].eval();self.runtime=None

    def get_runtime(self):
        if self.runtime is None:
            from clickagent.conditioned_features import ConditionedFeatureRuntime
            self.runtime=ConditionedFeatureRuntime(self.config)
            write(OUT/'current_qwen_runtime.json',self.runtime.runtime_metadata)
        return self.runtime

    def view(self,case,transform):
        require(set(case)==set(PUBLIC),'Extractor accepts public fields only')
        ident=digest(dict(case=case,transform=transform,contract=self.contract))
        p=OUT/'raw'/f'{ident}.npz'
        if not p.exists():
            require(sha(case['image'])==case['image_sha256'],'Image bytes changed')
            with Image.open(case['image']) as opened:image=opened.convert('RGB')
            require(list(image.size)==case['image_size'],'Image size mismatch')
            if transform is not None:
                crop=image.crop(transform['box']).resize(tuple(transform['crop_size']),Image.Resampling.LANCZOS);image.close();image=crop
            try:arrays,meta=self.get_runtime().infer_spatial_features(image,case['instruction'])
            finally:image.close()
            meta.update(sample_id=case['sample_id'],image_sha256=case['image_sha256'],instruction_sha256=hashlib.sha256(case['instruction'].encode()).hexdigest(),transform=transform,input_identity=ident)
            npz(p,metadata=np.asarray(json.dumps(meta,ensure_ascii=False)),**arrays)
        with np.load(p,allow_pickle=False) as z:
            meta=json.loads(str(z['metadata']));arrays={k:z[k].copy() for k in ('target_attention','target_hidden','image_hidden','vision_hidden')}
        require(meta['input_identity']==ident and meta['own_token_forward'] and meta['all_backbone_parameters_frozen'] and meta['ground_truth_in_model_input'] is False,'Raw metadata invalid')
        return arrays,meta,dict(feature_path=str(p),feature_sha256=sha(p),transform=transform)

    def case(self,public):
        require(set(public)==set(PUBLIC),'Labels in extraction input')
        path=OUT/f'pipeline/{key(public["sample_id"])}.json'
        if path.exists():
            pipe=read(path);require(pipe['complete'] and pipe['input_identity']==digest(public),'Cached pipeline drift')
            return pipe
        started=time.perf_counter(); arrays,meta,g=self.view(public,None)
        result=predict_raw(self.proposer,arrays,meta,public['image_size'],topk=64)
        views=[]
        for crop in plan_multiregion_crops(result,public['image_size']):
            arrays,meta,v=self.view(public,crop['transform'])
            decoded=predict_raw(self.proposer,arrays,meta,crop['transform']['crop_size'],topk=64)
            views.append(dict(v,region_index=crop['region_index'],result=decoded))
        pipe=dict(complete=True,sample_id=public['sample_id'],input_identity=digest(public),global_feature_path=g['feature_path'],
            global_feature_sha256=g['feature_sha256'],global_result=result,local_views=views,pools=build_c1_c3_pools(result,views),
            proposer_role='full',proposer_sha256=sha(V12/'proposers/full.pt'),ground_truth_in_model_inputs=False,elapsed_seconds=time.perf_counter()-started)
        write(path,pipe);return pipe


def extract_features(cohort,limit=None):
    require(cohort in ('new_train','new_reserve'),'Unknown cohort')
    if cohort=='new_reserve': require((OUT/'models/frozen.json').exists(),'Freeze models before reserve extraction')
    manifest=DATA/('train.jsonl' if cohort=='new_train' else 'validation_reserve.jsonl')
    cases=sorted(rows(manifest),key=lambda c:key(c['sample_id']));ip=OUT/f'features/{cohort}/index.json'
    if ip.exists():
        ix=read(ip);require(ix['complete'] and ix['manifest_sha256']==sha(manifest),'Feature index drift');return
    extractor=Extractor();entries=[];started=time.perf_counter();computed=0
    selected=cases if limit is None else cases[:limit]
    status('extracting_'+cohort,completed=0,total=len(cases),selected=len(selected))
    for i,c in enumerate(selected,1):
        dest=OUT/f'features/{cohort}/cases/{key(c["sample_id"])}.json'
        if dest.exists():
            e=read(dest)
            require(e['case_identity']==digest(c),'Feature case identity drift')
            for p,h in (('path','sha256'),('labels_path','labels_sha256'),('state_path','state_sha256'),('mask_path','mask_sha256')):
                require(sha(e[p])==e[h],'Completed feature changed')
            entries.append(e)
        else:
            public={k:c[k] for k in PUBLIC};pipe=extractor.case(public)
            base=build_base(pipe,c['image_size']);points=base['points'];features=[];masks=[]
            views=[dict(feature_path=pipe['global_feature_path'],feature_sha256=pipe['global_feature_sha256'],transform=None)]+pipe['local_views']
            for v in views:
                require(sha(v['feature_path'])==v['feature_sha256'],'Raw bytes drift')
                with np.load(v['feature_path'],allow_pickle=False) as z:
                    meta=json.loads(str(z['metadata']));arrays={k:z[k].copy() for k in ('target_attention','target_hidden','image_hidden')}
                f,m=state_features(arrays,meta,points,c['image_size'],v['transform']);features.append(f);masks.append(m)
            tmax=max(1,*(f.shape[1] for f in features));state=np.zeros((len(points),4,tmax,STATE_DIM),np.float16);mask=np.zeros((len(points),4,tmax),bool)
            for j,(f,m) in enumerate(zip(features,masks)):
                t=f.shape[1];state[:,j,:t]=f;mask[:,j,:t]=m[:,None]
            # Geometry is consumed only AFTER the entire public-only pipeline.
            labels=np.asarray([point_in_geometry((p/np.asarray(c['image_size'])).tolist(),c['target_geometry']) for p in points],np.float32)
            bp=OUT/f'features/{cohort}/arrays/{key(c["sample_id"])}.npz';npz(bp,**base)
            lp=bp.with_suffix('.labels.npy');sp=bp.with_suffix('.state.npy');mp=bp.with_suffix('.mask.npy')
            for p,v in ((lp,labels),(sp,state),(mp,mask)):
                temp=p.with_suffix(p.suffix+'.tmp')
                with temp.open('wb') as stream:np.save(stream,v,allow_pickle=False)
                replace(temp,p)
            e=dict(sample_id=c['sample_id'],split='train' if cohort=='new_train' else 'val',role=cohort,source=c['dataset'],group=c['pixel_sha256'],
                source_group=c['source_group'],case_identity=digest(c),n=len(points),matched_count=len(base['baseline']),path=str(bp),sha256=sha(bp),
                labels_path=str(lp),labels_sha256=sha(lp),state_path=str(sp),state_sha256=sha(sp),mask_path=str(mp),mask_sha256=sha(mp),
                pipeline_path=str(OUT/f'pipeline/{key(c["sample_id"])}.json'),pipeline_sha256=sha(OUT/f'pipeline/{key(c["sample_id"])}.json'))
            write(dest,e);entries.append(e);computed+=1
        if i%10==0 or i==len(selected):
            status('extracting_'+cohort,completed=i,total=len(cases),newly_computed=computed,elapsed_seconds=time.perf_counter()-started,
                last_sample_id=c['sample_id'],mean_new_case_seconds=(time.perf_counter()-started)/computed if computed else None)
    if len(selected)==len(cases):
        write(ip,dict(complete=True,cases=entries,n=len(entries),manifest_sha256=sha(manifest),protocol_sha256=sha(OUT/'protocol.json'),
            raw_forwards_new_only=True,ground_truth_in_model_input=False,elapsed_seconds=time.perf_counter()-started))
    else:write(OUT/f'features/{cohort}/probe.json',dict(complete=False,selected=len(entries),total=len(cases),cases=entries))
    del extractor;gc.collect()
    if torch.cuda.is_available():torch.cuda.empty_cache()


def freeze_models():
    receipts={}
    for regime,variants in (('old_matched',('aligned',)),('expanded',('rank','aligned','permuted'))):
        for variant in variants:
            for seed in SEEDS:
                name=model_name(regime,variant,seed);r=read(OUT/f'models/{name}.json')
                require(r['complete'] and r['steps']==STEPS and sha(r['checkpoint_path'])==r['checkpoint_sha256'],'Incomplete model')
                receipts[name]=dict(path=r['checkpoint_path'],sha256=r['checkpoint_sha256'],variant=variant)
    for variant in ('rank','aligned','permuted'):
        for s in (1,2):
            r=read(OLD/f'models/{variant}_s{s}.json');receipts[f'old8_{variant}_s{s}']=dict(path=r['checkpoint_path'],sha256=r['checkpoint_sha256'],variant=variant)
    value=dict(complete=True,models=receipts,protocol_sha256=sha(OUT/'protocol.json'),source_identity=identity())
    p=OUT/'models/frozen.json'
    if p.exists():require(read(p)==value,'Model seal drift')
    else:write(p,value)
    status('all_models_frozen',models=len(receipts),new_models=8)


def compare(a,b,entries):
    result=paired(a,b,entries,bootstrap=True)
    if entries[0]['role']=='new_reserve':
        by_group=[dict(e,group=e['source_group']) for e in entries]
        result['source_group_ci95']=paired(a,b,by_group,bootstrap=True)['ci95']
    result['uncertainty']='Fixed trained-model paired source-stratified RGB bootstrap2000; source-group interval additionally on new development reserve. No final independent-test claim.'
    return result


@torch.inference_mode()
def evaluate(cohort,device):
    frozen=read(OUT/'models/frozen.json');require(frozen['complete'],'Models not frozen')
    entries=old_entries('val') if cohort=='old_val' else read(OUT/'features/new_reserve/index.json')['cases']
    records=load_records(entries);vectors={};details={r['sample_id']:dict(sample_id=r['sample_id'],source=r['source'],group=r['group'],source_group=r['source_group'],methods={}) for r in records}
    status('evaluating_'+cohort,models=len(frozen['models']),queries=len(records))
    for name,e in frozen['models'].items():
        require(sha(e['path'])==e['sha256'],'Frozen checkpoint drift');saved=torch.load(e['path'],weights_only=True,map_location='cpu')
        model=RetentionModel().to(device);model.load_state_dict(saved['state_dict']);model.eval();mean,std=saved['mean'].numpy(),saved['std'].numpy()
        metrics={k:[] for k in ('old_k16_top1','old_k16_top4','union_top1')}
        for i,r in enumerate(records,1):
            p=OUT/f'predictions/{cohort}/{name}/{key(r["sample_id"])}.npz'
            if p.exists():
                with np.load(p,allow_pickle=False) as z:
                    require(str(z['checkpoint_sha256'])==e['sha256'] and str(z['base_sha256'])==r['sha256'],'Cached prediction drift');scores=z['scores'].copy()
            else:
                if r['n']:
                    inputs,_=tensors([r],mean,std,e['variant'],device);scores=model(**inputs).float().cpu().numpy()
                else:scores=np.empty(0,np.float32)
                npz(p,scores=scores,checkpoint_sha256=np.asarray(e['sha256']),base_sha256=np.asarray(r['sha256']))
            require(scores.shape==(r['n'],) and np.isfinite(scores).all(),'Bad scores')
            old=np.sort(r['baseline']);order=old[np.argsort(-scores[old],kind='stable')];chosen=int(order[0]) if len(order) else None
            union=int(scores.argmax()) if len(scores) else None
            value=dict(old_k16_point=r['points'][chosen].tolist() if chosen is not None else None,
                old_k16_top1=bool(chosen is not None and r['labels'][chosen]),old_k16_top4=bool(r['labels'][order[:4]].any()),
                union_top1=bool(union is not None and r['labels'][union]))
            details[r['sample_id']]['methods'][name]=value
            for k in metrics:metrics[k].append(value[k])
            if i%200==0:print(json.dumps(dict(stage='evaluation',cohort=cohort,model=name,i=i,n=len(records))),flush=True)
        vectors[name]=metrics;del model
        status('evaluating_'+cohort,models_completed=len(vectors),models_total=len(frozen['models']),queries=len(records),last_model=name)
    methods={name:dict(old_k16_top1_hits=sum(v['old_k16_top1']),old_k16_top4_coverage=sum(v['old_k16_top4']),union_top1_hits=sum(v['union_top1']),
        accuracy=sum(v['old_k16_top1'])/len(records)) for name,v in vectors.items()}
    comparisons={}
    for s in (1,2):
        a=f'expanded_aligned_s{s}'
        for b in (f'old8_aligned_s{s}',f'old_matched_aligned_s{s}',f'expanded_rank_s{s}',f'expanded_permuted_s{s}'):
            comparisons[a+' vs '+b]=compare(vectors[a]['old_k16_top1'],vectors[b]['old_k16_top1'],entries)
    result=dict(complete=True,cohort=cohort,n=len(records),rgb_groups=len({r['group'] for r in records}),source_groups=len({(r['source'],r['source_group']) for r in records}),
        methods=methods,comparisons=comparisons,old_k16_coverage=sum(bool(r['labels'][r['baseline']].any()) for r in records),
        union_coverage=sum(bool(r['labels'].any()) for r in records),empty_candidate_queries=sum(not r['n'] for r in records),
        by_source={s:dict(n=sum(r['source']==s for r in records),methods={name:sum(hit for hit,r in zip(v['old_k16_top1'],records) if r['source']==s) for name,v in vectors.items()}) for s in sorted({r['source'] for r in records})},
        model_seal_sha256=sha(OUT/'models/frozen.json'),scope='Development-only static point evaluation, no GUI or actual clicks; all samples including no proposals remain in denominator.')
    write(OUT/f'evaluation/{cohort}/details.json',list(details.values()));write(OUT/f'evaluation/{cohort}/results.json',result);render()
    del records;gc.collect()


def complete():
    verify();legacy_hashes()
    for name,expected in read(V13/'frozen.json')['files'].items():require(sha(name)==expected,'V13 sealed drift: '+name)
    paths=[OUT/'protocol.json',OUT/'models/frozen.json',OUT/'features/new_train/index.json',OUT/'features/new_reserve/index.json']
    paths += [OUT/f'evaluation/{c}/{f}.json' for c in ('old_val','new_reserve') for f in ('results','details')]
    require(all(read(p).get('complete',True) for p in paths),'Incomplete stage')
    write(OUT/'completion.json',dict(complete=True,completed_at_utc=now(),artifact_sha256={str(p):sha(p) for p in paths},source_identity=identity(),legacy_unchanged=True,
        independent_final_test=False,training_steps_per_new_model=STEPS,new_models=8))
    status('training_and_development_evaluation_complete',new_models=8,evaluated_models=14,old_val_queries=172,new_reserve_queries=1439)
    state=read(OUT/'status.json');state['complete']=True;write(OUT/'status.json',state);render()


def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=('register','old-controls','extract-train','train-expanded','freeze','extract-reserve','evaluate','all'),required=True)
    p.add_argument('--device',default='cuda:0');p.add_argument('--limit',type=int);args=p.parse_args()
    os.chdir(ROOT);torch.set_num_threads(4);register();verify()
    try:
        if args.stage in ('old-controls','all'):
            for seed in SEEDS:train_one('old_matched','aligned',seed,args.device)
        if args.stage in ('extract-train','all'):extract_features('new_train',args.limit)
        if args.stage in ('train-expanded','all'):
            for variant in ('rank','aligned','permuted'):
                for seed in SEEDS:train_one('expanded',variant,seed,args.device)
        if args.stage in ('freeze','all'):freeze_models()
        if args.stage in ('extract-reserve','all'):extract_features('new_reserve',args.limit)
        if args.stage in ('evaluate','all'):
            for cohort in ('old_val','new_reserve'):evaluate(cohort,args.device)
            complete()
    except BaseException as exc:
        prior=read(OUT/'status.json') if (OUT/'status.json').exists() else {}
        write(OUT/f'failures/{datetime.now().strftime("%Y%m%d_%H%M%S")}.json',dict(error=repr(exc),traceback=traceback.format_exc(),prior=prior))
        status('failed',error=repr(exc),previous_phase=prior.get('phase'),resume_command='python scripts/run_quality_readout_v15.py --stage all')
        raise


if __name__=='__main__':main()
