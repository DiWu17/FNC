"""Registered 2973-query scaling run; legacy sources and caches are read-only.

Runs selected extraction, paired controls, three seeded expanded fits, model
freeze, then selected new-development extraction and evaluation automatically.
"""
from __future__ import annotations

import os
os.environ.setdefault('PYTHONDONTWRITEBYTECODE', '1')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
from collections import Counter
from dataclasses import asdict, replace as dataclass_replace
from datetime import datetime, timezone
import gc
import hashlib
import html
import json
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
import run_quality_readout_v15 as base
from clickagent.evidence_retention import RetentionModel, retention_loss, require
from run_evidence_retention_v14 import tensors, paired

OUT = ROOT/'runs/quality_readout_v15_3k'
SOURCES = ('GroundCUA', 'ShowUI-web', 'AMEX')
SEEDS = (20261311, 20261312, 20261313)
TRAIN_PER_SOURCE = 700
RESERVE_PER_SOURCE = 100
TOTAL = 873 + 3*TRAIN_PER_SOURCE
STEPS = 8*((TOTAL+7)//8)
read, write, sha, key, digest = base.read, base.write, base.sha, base.key, base.digest


def select_subset(rows, per_source):
    result = []
    for source in SOURCES:
        group = sorted((r for r in rows if r['dataset'] == source), key=lambda r:key(r['sample_id']))
        require(len(group) >= per_source, 'Insufficient fixed source cohort')
        result.extend(group[:per_source])
    require(len({r['sample_id'] for r in result}) == len(result), 'Repeated sample ID')
    return sorted(result, key=lambda r:key(r['sample_id']))


def source_identity():
    original = read(base.OUT/'protocol.json')['source_identity']
    require(original == base.identity(), 'Original V15 scientific source drift')
    extra = (Path(__file__), ROOT/'tests/test_quality_readout_v15_3k.py')
    return dict(original, **{str(p.relative_to(ROOT)):sha(p) for p in extra})


def render():
    s = read(OUT/'status.json') if (OUT/'status.json').exists() else {}
    names = {'registered':'样本与方案已固定', 'extracting_new_train':'补齐新增训练特征',
        'training':'训练评分器与旧数据对照', 'models_frozen':'全部模型已冻结',
        'extracting_new_reserve':'提取新来源开发验证特征', 'evaluating_old_val':'评估旧验证集',
        'evaluating_new_reserve':'评估新来源开发验证集', 'complete':'训练与开发验证已完成', 'failed':'运行遇到错误'}
    body = ['<h1>约3000条数据：定位评分器训练与验证</h1>',
        '<p>训练总量 <b>2973条＝原873＋新增2100</b>；GroundCUA、ShowUI-web、AMEX各700条。22,050参数评分器，3个种子。</p>',
        '<p>每个扩容模型固定2976步；同时训练旧数据同步数对照。旧8轮参考另列。全部模型冻结后才处理新来源开发验证样本。</p>',
        f'<div class="card"><h2>{html.escape(names.get(s.get("phase"), "准备中"))}</h2>']
    phase = s.get('phase', '')
    if phase.startswith('extracting_'):
        n, total = s.get('completed',0), s.get('total',1)
        body.append(f'<p><b>{n} / {total} 条</b> · 复用 {s.get("reused",0)} · 本轮新算 {s.get("computed",0)}</p><progress max="{total}" value="{n}"></progress>')
        if s.get('remaining_hours') is not None:
            body.append(f'<p>按本轮已完成的新样本粗估，本阶段剩余 {s["remaining_hours"]:.1f} 小时；后续训练和验证另计。</p>')
    elif phase == 'training':
        body.append(f'<p>{html.escape(s.get("model",""))}：{s.get("step",0)} / {s.get("total_steps",STEPS)} 步</p>')
    elif phase.startswith('evaluating_'):
        body.append(f'<p>已评估 {s.get("models_completed",0)} / 9 个模型。</p>')
    body.append(f'<p class="muted">更新时间（UTC）：{html.escape(s.get("updated_at_utc",""))}；页面每30秒刷新。</p></div>')
    for cohort, title in [('old_val','旧验证集：172条'), ('new_reserve','新来源开发验证：300条')]:
        p = OUT/f'evaluation/{cohort}/results.json'
        if not p.exists(): continue
        result = read(p)
        body.append(f'<h2>{title}</h2><table><tr><th>训练方案</th><th>S1命中</th><th>S2命中</th><th>S3命中</th><th>平均准确率</th></tr>')
        for arm, title2 in [('old8','原873条，8轮'),('old_matched','原873条，2976步'),('expanded','2973条，2976步')]:
            v = [result['methods'][f'{arm}_s{s}']['top1_hits'] for s in (1,2,3)]
            body.append(f'<tr><td>{title2}</td>'+''.join(f'<td>{x}/{result["n"]}</td>' for x in v)+f'<td>{100*np.mean(v)/result["n"]:.2f}%</td></tr>')
        body.append('</table>')
    body.extend(['<div class="card"><b>如何理解本轮</b><p>评价同一原K16候选中的最终Top1几何命中。新来源按导入时分组保留的开发集取样，属于开发评估；旧验证有历史复用。没有真实点击或行为分类训练，也不构成论文最终独立测试。</p><p>原1.25万条全量任务仍暂停；本轮只补选定2100条训练与300条验证，既有缓存保留。</p></div>',
        '<p><a href="protocol.json">固定方案</a> · <a href="data/cohort_summary.json">样本统计</a> · <a href="logs/events.jsonl">运行日志</a> · <a href="status.json">技术状态</a> · <a href="../quality_readout_v15_pilot/report.html">前一轮小试</a></p>'])
    page = '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta http-equiv="refresh" content="30"><title>ClickAgent 3000条实验</title><style>body{max-width:1100px;margin:32px auto;padding:0 22px;font:16px/1.7 Segoe UI,Microsoft YaHei,sans-serif;background:#f3f6fb;color:#24364b}.card{background:white;padding:20px;border-radius:12px;margin:20px 0}.muted{color:#63758b;font-size:14px}table{width:100%;border-collapse:collapse;background:white}th,td{padding:9px;border-bottom:1px solid #ddd;text-align:left}progress{width:100%;height:22px}a{color:#176e9e}</style><body>'+'\n'.join(body)+'</body></html>'
    p = OUT/'report.html'; temp = p.with_suffix('.html.tmp');temp.write_text(page,encoding='utf-8');base.replace(temp,p)


def status(phase, **extra):
    s = dict(phase=phase, complete=phase=='complete', pid=os.getpid(),
        updated_at_utc=datetime.now(timezone.utc).isoformat(), **extra)
    write(OUT/'status.json',s)
    log = OUT/'logs/events.jsonl';log.parent.mkdir(parents=True,exist_ok=True)
    with log.open('a',encoding='utf-8') as stream:stream.write(json.dumps(s,ensure_ascii=False)+'\n')
    render();print(json.dumps(s,ensure_ascii=False),flush=True)


def register():
    p = OUT/'protocol.json'
    if p.exists():
        spec = read(p);require(spec['source_identity']==source_identity(),'3k source drift')
        for filename, expected in spec['input_sha256'].items():require(sha(filename)==expected,'Frozen input drift: '+filename)
        return
    prep_path = base.OUT/'analysis/data_suitability/qc1_candidate/preparation.json'
    prep = read(prep_path); inputs = {str(prep_path):sha(prep_path), **base.legacy_hashes()}
    selected = {}
    for cohort, name, count in [('new_train','train.jsonl',TRAIN_PER_SOURCE),('new_reserve','validation_reserve.jsonl',RESERVE_PER_SOURCE)]:
        info = prep['candidate_manifests'][name];require(sha(info['path'])==info['sha256'],'Clean manifest changed')
        inputs[info['path']] = info['sha256']; selected[cohort] = select_subset(base.rows(info['path']),count)
    old = base.old_entries('train'); oldval = base.old_entries('val')
    require(len(old)==873 and len(oldval)==172 and TOTAL==2973 and STEPS==2976,'Unexpected experiment size')
    train_rgb = {r['group'] for r in old}|{r['pixel_sha256'] for r in selected['new_train']}
    val_rgb = {r['group'] for r in oldval}|{r['pixel_sha256'] for r in selected['new_reserve']}
    require(not train_rgb&val_rgb,'Train/validation RGB overlap')
    a = {(r['dataset'],r['source_group']) for r in selected['new_train']}
    b = {(r['dataset'],r['source_group']) for r in selected['new_reserve']}
    require(not a&b,'New import source-group overlap')
    pilot = base.rows(ROOT/'runs/quality_readout_v15_pilot/data/add90.jsonl')
    require({r['sample_id'] for r in pilot}<={r['sample_id'] for r in selected['new_train']},'Not nested with prior pilot')
    summary = {'old_train':dict(queries=len(old),images=len({r['group'] for r in old}))}
    for cohort, cases in selected.items():
        path = OUT/f'data/{cohort}.jsonl';path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(''.join(json.dumps(c,ensure_ascii=False)+'\n' for c in cases),encoding='utf-8');inputs[str(path)] = sha(path)
        summary[cohort] = dict(queries=len(cases),images=len({c['pixel_sha256'] for c in cases}),
            by_source=dict(Counter(c['dataset'] for c in cases)),
            source_groups={s:len({c['source_group'] for c in cases if c['dataset']==s}) for s in SOURCES})
    write(OUT/'data/old_train_index.json',old);write(OUT/'data/cohort_summary.json',summary)
    for path in (OUT/'data/old_train_index.json',OUT/'data/cohort_summary.json',base.OUT/'protocol.json',
        base.V12/'runtime_identity.json',base.V12/'proposers/full.pt',ROOT/'runs/quality_readout_v15_pilot/completion.json'):
        inputs[str(path)] = sha(path)
    for s in (1,2):
        for ext in ('.pt','.json'):
            path = base.OLD/f'models/aligned_s{s}{ext}';inputs[str(path)] = sha(path)
    spec = dict(created_at_utc=datetime.now(timezone.utc).isoformat(),source_identity=source_identity(),input_sha256=inputs,
        question='2973条训练数据能否在相同模型与优化步数下改善定位评分，并在三个种子及新来源开发验证中重复？',
        queries=dict(old_train=873,new_train=2100,total_train=TOTAL,old_val=172,new_development=300),seeds=list(SEEDS),parameters=22050,
        selection='First700 train and first100 preheld development queries per source in sample-ID SHA order, after existing placeholder-only QC. Fixed before new feature outcomes; no cache, candidate-reachability or model-success filter. Train nested with previous90-query pilot.',
        upstream=read(base.OUT/'protocol.json')['upstream'],
        extraction='Same public-only V15 Extractor mathematics, raw six-array retention, unchanged crops/proposer/32 base+294 state. Original V15 complete cases/raw/pipelines are read-only authenticated cache references; new files remain inside this3k run. No quantization, reduced precision changes, token/head dropping or changed inference batch.',
        arms={'old8':'Archived V14 aligned S1/S2; train S3 for856 updates (8 old-data epochs). Secondary equal-exposure reference.',
            'old_matched':'Old873 only,2976 updates, all3 seeds.',
            'expanded':'Old873+2100 new,2976 updates, all3 seeds.'},
        optimization=dict(steps=STEPS,old8_steps=856,batch_queries=8,lr=.0002,weight_decay=.02,gradient_clip=1.,dropout=0,
            nominal_passes=8,normalization='Train-regime-only candidate-base mean/std; no development input',
            loss='Unchanged full-union multipositive listwise/null +.25 balanced BCE',
            empty_queries='Keep selected cohorts fixed; skip zero-candidate fitting records, retain every evaluation query in denominators',
            stopping='Fixed final step, no validation or seed selection; optimizer/RNG/cursor resume every250 updates'),
        workflow=['selected training features','all7 new model fits','freeze all9 checkpoints','selected new development features','both development evaluations','completion integrity audit'],
        primary='Forced Top1 on same actual originalK16, canonical union-index tie order. Expanded vs old_matched on old_val172 and new_development300; all3 seeds and their mean reported, no pooled344/516 pseudo-independent sample claim.',
        secondary=['Expanded vs old8 eachseed','Source-macro accuracy','OriginalK16 Top4 coverage','Whole-union Top1 and coverage separately'],
        uncertainty='Eachseed paired source-stratified RGB-cluster bootstrap2000; source-group bootstrap additionally on new development. Seed-mean paired bootstrap averages predictions across fixed seeds before cluster resampling; conditional on these3 fits, not a training-seed confidence interval.',
        isolation='Train/development RGB and new-import source groups disjoint. All9 checkpoints frozen before selected new-development feature extraction or prediction. No historical192 benchmark predictions; no reserve-based checkpoint, seed or mixture selection.',
        limits=['Old validation historically reused; new development groups are held within this import, not globally unseen software.',
            'AMEX source groups are episode proxies, not app IDs; only3 held Web source groups may limit group-interval precision.',
            'Changing query count also changes source diversity and mixture; cannot attribute all gains to quantity alone.',
            'Placeholder cleanup complete; separate90-query manual semantic audit remains pending, so no fully manually certified-label claim.',
            'Frozen pretraining overlap unknown; image-level dedup cannot prove semantic independence.',
            'Only static localization scoring; no execution-action classifier or real clicks.'])
    for name in spec['source_identity']:
        dest = OUT/'source_archive'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,dest)
    write(p,spec);status('registered',queries=spec['queries'],steps=STEPS,seeds=list(SEEDS))


class Extractor(base.Extractor):
    def __init__(self):
        super().__init__()
        self.config = dataclass_replace(self.config,offload_folder=str(OUT/'offload'))
        cfg = asdict(self.config);cfg.pop('offload_folder')
        require(json.loads(json.dumps(cfg))==self.contract,'3k extraction contract changed')

    def get_runtime(self):
        if self.runtime is None:
            from clickagent.conditioned_features import ConditionedFeatureRuntime
            self.runtime = ConditionedFeatureRuntime(self.config)
            write(OUT/'current_qwen_runtime.json',self.runtime.runtime_metadata)
        return self.runtime

    def view(self, case, transform):
        require(set(case)==set(base.PUBLIC),'Extractor accepts public fields only')
        ident = digest(dict(case=case,transform=transform,contract=self.contract))
        shared = base.OUT/'raw'/f'{ident}.npz'; path = shared if shared.exists() else OUT/'raw'/f'{ident}.npz'
        if not path.exists():
            require(sha(case['image'])==case['image_sha256'],'Image bytes changed')
            with Image.open(case['image']) as opened:image = opened.convert('RGB')
            require(list(image.size)==case['image_size'],'Image size mismatch')
            if transform is not None:
                cropped = image.crop(transform['box']).resize(tuple(transform['crop_size']),Image.Resampling.LANCZOS);image.close();image=cropped
            try:arrays,meta = self.get_runtime().infer_spatial_features(image,case['instruction'])
            finally:image.close()
            meta.update(sample_id=case['sample_id'],image_sha256=case['image_sha256'],instruction_sha256=hashlib.sha256(case['instruction'].encode()).hexdigest(),transform=transform,input_identity=ident)
            base.npz(path,metadata=np.asarray(json.dumps(meta,ensure_ascii=False)),**arrays)
        with np.load(path,allow_pickle=False) as z:
            meta = json.loads(str(z['metadata']));arrays={k:z[k].copy() for k in ('target_attention','target_hidden','image_hidden','vision_hidden')}
        require(meta['input_identity']==ident and meta['own_token_forward'] and meta['all_backbone_parameters_frozen'] and meta['ground_truth_in_model_input'] is False,'Raw metadata invalid')
        return arrays,meta,dict(feature_path=str(path),feature_sha256=sha(path),transform=transform)

    def case(self, public):
        require(set(public)==set(base.PUBLIC),'Labels in extraction input')
        own = OUT/f'pipeline/{key(public["sample_id"])}.json'; shared=base.OUT/f'pipeline/{key(public["sample_id"])}.json'
        path = own if own.exists() else shared
        if path.exists():
            pipe=read(path);require(pipe['complete'] and pipe['input_identity']==digest(public),'Cached pipeline drift')
            for v in [dict(feature_path=pipe['global_feature_path'],feature_sha256=pipe['global_feature_sha256'])]+pipe['local_views']:
                require(sha(v['feature_path'])==v['feature_sha256'],'Cached raw drift')
            return pipe, path
        started=time.perf_counter();arrays,meta,g=self.view(public,None)
        result=base.predict_raw(self.proposer,arrays,meta,public['image_size'],topk=64);views=[]
        for crop in base.plan_multiregion_crops(result,public['image_size']):
            arrays,meta,v=self.view(public,crop['transform'])
            decoded=base.predict_raw(self.proposer,arrays,meta,crop['transform']['crop_size'],topk=64)
            views.append(dict(v,region_index=crop['region_index'],result=decoded))
        pipe=dict(complete=True,sample_id=public['sample_id'],input_identity=digest(public),global_feature_path=g['feature_path'],global_feature_sha256=g['feature_sha256'],global_result=result,
            local_views=views,pools=base.build_c1_c3_pools(result,views),proposer_role='full',proposer_sha256=sha(base.V12/'proposers/full.pt'),ground_truth_in_model_inputs=False,elapsed_seconds=time.perf_counter()-started)
        write(own,pipe);return pipe,own


def authenticate_entry(entry, case):
    require(entry['case_identity']==digest(case),'Cached row identity changed')
    for field,h in [('path','sha256'),('labels_path','labels_sha256'),('state_path','state_sha256'),('mask_path','mask_sha256'),('pipeline_path','pipeline_sha256')]:
        require(sha(entry[field])==entry[h],'Cached feature drift: '+entry[field])


def compact(case, pipe, pipeline_path):
    b=base.build_base(pipe,case['image_size']);features=[];masks=[]
    views=[dict(feature_path=pipe['global_feature_path'],feature_sha256=pipe['global_feature_sha256'],transform=None)]+pipe['local_views']
    for v in views:
        require(sha(v['feature_path'])==v['feature_sha256'],'Raw bytes drift')
        with np.load(v['feature_path'],allow_pickle=False) as z:
            meta=json.loads(str(z['metadata']));arrays={k:z[k].copy() for k in ('target_attention','target_hidden','image_hidden')}
        f,m=base.state_features(arrays,meta,b['points'],case['image_size'],v['transform']);features.append(f);masks.append(m)
    tmax=max(1,*(f.shape[1] for f in features));state=np.zeros((len(b['points']),4,tmax,base.STATE_DIM),np.float16);mask=np.zeros((len(b['points']),4,tmax),bool)
    for j,(f,m) in enumerate(zip(features,masks)):
        t=f.shape[1];state[:,j,:t]=f;mask[:,j,:t]=m[:,None]
    labels=np.asarray([base.point_in_geometry((p/np.asarray(case['image_size'])).tolist(),case['target_geometry']) for p in b['points']],np.float32)
    return b,labels,state,mask


def extract(cohort, limit=None):
    if cohort=='new_reserve':frozen_models()
    manifest=OUT/f'data/{cohort}.jsonl';cases=base.rows(manifest);index=OUT/f'features/{cohort}/index.json'
    if index.exists():
        ix=read(index);require(ix['complete'] and ix['manifest_sha256']==sha(manifest),'Feature index drift');return
    selected=cases if limit is None else cases[:limit]
    entries=[];extractor=None;reused=computed=0;new_seconds=0.;started=time.perf_counter()
    status('extracting_'+cohort,completed=0,total=len(cases),reused=0,computed=0)
    for i,c in enumerate(selected,1):
        dest=OUT/f'features/{cohort}/cases/{key(c["sample_id"])}.json';shared=base.OUT/f'features/{cohort}/cases/{key(c["sample_id"])}.json'
        old=dest if dest.exists() else shared
        if old.exists():
            e=read(old);authenticate_entry(e,c);reused+=1
            if old!=dest:write(dest,dict(e,reused_from=str(old),reused_entry_sha256=sha(old)))
            e=read(dest)
        else:
            start_case=time.perf_counter()
            if extractor is None:extractor=Extractor()
            pipe,pp=extractor.case({k:c[k] for k in base.PUBLIC});b,labels,state,mask=compact(c,pipe,pp)
            bp=OUT/f'features/{cohort}/arrays/{key(c["sample_id"])}.npz';base.npz(bp,**b)
            lp=bp.with_suffix('.labels.npy');sp=bp.with_suffix('.state.npy');mp=bp.with_suffix('.mask.npy')
            for p,v in [(lp,labels),(sp,state),(mp,mask)]:
                temp=p.with_suffix(p.suffix+'.tmp')
                with temp.open('wb') as stream:np.save(stream,v,allow_pickle=False)
                base.replace(temp,p)
            e=dict(sample_id=c['sample_id'],split='train' if cohort=='new_train' else 'val',role=cohort,source=c['dataset'],group=c['pixel_sha256'],source_group=c['source_group'],
                case_identity=digest(c),n=len(b['points']),matched_count=len(b['baseline']),path=str(bp),sha256=sha(bp),labels_path=str(lp),labels_sha256=sha(lp),
                state_path=str(sp),state_sha256=sha(sp),mask_path=str(mp),mask_sha256=sha(mp),pipeline_path=str(pp),pipeline_sha256=sha(pp))
            write(dest,e);computed+=1;new_seconds+=time.perf_counter()-start_case
        entries.append(e)
        if i%10==0 or i==len(selected):
            status('extracting_'+cohort,completed=i,total=len(cases),reused=reused,computed=computed,
                elapsed_seconds=time.perf_counter()-started,mean_new_case_seconds=new_seconds/computed if computed else None,
                remaining_hours=(len(cases)-i)*new_seconds/computed/3600 if computed else None,last_sample_id=c['sample_id'])
    if len(selected)==len(cases):write(index,dict(complete=True,cases=entries,n=len(entries),manifest_sha256=sha(manifest),protocol_sha256=sha(OUT/'protocol.json')))
    else:write(OUT/f'features/{cohort}/probe.json',dict(complete=False,selected=len(entries),total=len(cases),cases=entries))
    del extractor;gc.collect()
    if torch.cuda.is_available():torch.cuda.empty_cache()


def train_one(arm, records, seed, steps, device):
    tag=f'{arm}_s{SEEDS.index(seed)+1}';receipt=OUT/f'models/{tag}.json'
    if receipt.exists():
        e=read(receipt);require(e['complete'] and e['steps']==steps and sha(e['checkpoint_path'])==e['checkpoint_sha256'],'Completed model drift');return
    fitting=[r for r in records if r['n']];x=np.concatenate([r['base'] for r in fitting]);mean=x.mean(0);std=x.std(0);std[std<1e-5]=1;del x
    torch.manual_seed(seed);np.random.seed(seed);model=RetentionModel().to(device)
    initial=digest({k:hashlib.sha256(v.detach().cpu().numpy().tobytes()).hexdigest() for k,v in model.state_dict().items()})
    if seed in SEEDS[:2]:require(initial==read(base.OLD/f'models/aligned_s{SEEDS.index(seed)+1}.json')['initial_state_sha256'],'Legacy initialization drift')
    paired_receipt=OUT/f'models/old_matched_s{SEEDS.index(seed)+1}.json'
    if arm=='expanded':require(initial==read(paired_receipt)['initial_state_sha256'],'Paired initializations differ')
    optimizer=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=.02)
    rp=OUT/f'models/{tag}.resume.pt';step=0;epoch=1;position=0;elapsed=0.;history=[]
    if rp.exists():
        saved=torch.load(rp,map_location='cpu',weights_only=True)
        require(saved['protocol_sha256']==sha(OUT/'protocol.json') and saved['entry_identity']==digest([{k:r[k] for k in ('sample_id','sha256','state_sha256','labels_sha256')} for r in records]),'Resume identity drift')
        model.load_state_dict(saved['state_dict']);optimizer.load_state_dict(saved['optimizer']);step=saved['step'];epoch=saved['epoch'];position=saved['position'];elapsed=saved['elapsed_seconds'];history=saved['history']
        torch.set_rng_state(saved['torch_rng'])
        if device.startswith('cuda'):torch.cuda.set_rng_state_all(saved['cuda_rng'])
    status('training',model=tag,step=step,total_steps=steps,queries=len(records),fitting_queries=len(fitting));start=time.perf_counter();losses=[];model.train()
    while step<steps:
        ix,epoch,position=base.next_batch(len(fitting),seed,epoch,position);batch=[fitting[i] for i in ix]
        inputs,lengths=tensors(batch,mean,std,'aligned',device);optimizer.zero_grad(set_to_none=True);scores=model(**inputs)
        loss=torch.stack([retention_loss(s,torch.as_tensor(r['labels'],device=device),model.null) for s,r in zip(scores.split(lengths),batch)]).mean()
        require(bool(torch.isfinite(loss)),'Nonfinite loss');loss.backward()
        require(all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters()),'Nonfinite gradient')
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();step+=1;losses.append(float(loss.detach()))
        if step%250==0 or step==steps:
            h=dict(step=step,epoch=epoch,position=position,train_loss=float(np.mean(losses)),elapsed_seconds=elapsed+time.perf_counter()-start);history.append(h);losses=[]
            rp.parent.mkdir(parents=True,exist_ok=True);temp=rp.with_suffix('.pt.tmp')
            torch.save(dict(state_dict=model.state_dict(),optimizer=optimizer.state_dict(),step=step,epoch=epoch,position=position,elapsed_seconds=h['elapsed_seconds'],history=history,
                protocol_sha256=sha(OUT/'protocol.json'),entry_identity=digest([{k:r[k] for k in ('sample_id','sha256','state_sha256','labels_sha256')} for r in records]),
                torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if device.startswith('cuda') else []),temp);base.replace(temp,rp)
            status('training',model=tag,total_steps=steps,**h)
    ckpt=OUT/f'models/{tag}.pt';temp=ckpt.with_suffix('.pt.tmp')
    torch.save(dict(state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},mean=torch.tensor(mean),std=torch.tensor(std),variant='aligned',seed=seed),temp);base.replace(temp,ckpt)
    write(receipt,dict(complete=True,name=tag,steps=steps,seed=seed,queries=len(records),fitting_queries=len(fitting),empty_skipped=len(records)-len(fitting),parameter_count=sum(p.numel() for p in model.parameters()),
        checkpoint_path=str(ckpt),checkpoint_sha256=sha(ckpt),initial_state_sha256=initial,history=history,protocol_sha256=sha(OUT/'protocol.json'),validation_used_for_fitting_or_selection=False))
    del model,optimizer
    if torch.cuda.is_available():torch.cuda.empty_cache()


def train_all(device):
    old=base.load_records(read(OUT/'data/old_train_index.json'));new=base.load_records(read(OUT/'features/new_train/index.json')['cases'])
    require(len(old)==873 and len(new)==2100,'Wrong training cohort')
    x=np.concatenate([r['base'] for r in old if r['n']]);mean=x.mean(0);std=x.std(0);std[std<1e-5]=1;del x
    for s in (1,2):
        ref=read(base.OLD/f'models/aligned_s{s}.json');saved=torch.load(ref['checkpoint_path'],weights_only=True,map_location='cpu')
        require(np.array_equal(mean,saved['mean'].numpy()) and np.array_equal(std,saved['std'].numpy()),'Old normalization replay differs')
        require(sha(ref['checkpoint_path'])==ref['checkpoint_sha256'],'Archived reference drift')
        write(OUT/f'models/old8_s{s}.json',dict(ref,steps=856,queries=873,reused_archived_reference=True))
    for seed in SEEDS:
        train_one('old_matched',old,seed,STEPS,device)
        if seed==SEEDS[2]:train_one('old8',old,seed,856,device)
        train_one('expanded',old+new,seed,STEPS,device)
    del old,new;gc.collect()


def freeze():
    models={}
    for arm in ('old8','old_matched','expanded'):
        for s in (1,2,3):
            name=f'{arm}_s{s}';e=read(OUT/f'models/{name}.json');require(e['complete'] and sha(e['checkpoint_path'])==e['checkpoint_sha256'],'Incomplete checkpoint')
            models[name]=dict(path=e['checkpoint_path'],sha256=e['checkpoint_sha256'],steps=e['steps'],queries=e['queries'],variant='aligned')
    v=dict(complete=True,models=models,protocol_sha256=sha(OUT/'protocol.json'),source_identity=source_identity())
    p=OUT/'models/frozen.json'
    if p.exists():require(read(p)==v,'Model freeze drift')
    else:write(p,v)
    status('models_frozen',models=9,new_models=7)


def frozen_models():
    p=OUT/'models/frozen.json'
    require(p.exists(),'Freeze all models before development extraction')
    v=read(p)
    require(v.get('complete') and len(v.get('models',{}))==9,'Incomplete model freeze')
    require(v['protocol_sha256']==sha(OUT/'protocol.json') and v['source_identity']==source_identity(),'Model freeze identity drift')
    for model in v['models'].values():require(sha(model['path'])==model['sha256'],'Frozen checkpoint drift')
    return v


def mean_seed_comparison(a,b,entries,group_field='group'):
    diff=np.mean(np.asarray(a,float)-np.asarray(b,float),axis=0)
    require(diff.shape==(len(entries),),'Seed-average shape mismatch')
    strata={}
    for i,e in enumerate(entries):strata.setdefault(e['source'],{}).setdefault(e[group_field],[]).append(i)
    groups=[list(g.values()) for g in strata.values()];rng=np.random.default_rng(20261401);values=[]
    for _ in range(2000):
        ix=[i for clusters in groups for j in rng.integers(0,len(clusters),len(clusters)) for i in clusters[j]]
        values.append(float(diff[ix].mean()))
    return dict(delta=float(diff.mean()),ci95=np.quantile(values,[.025,.975]).tolist(),
        uncertainty='Paired cluster bootstrap over seed-averaged predictions, conditional on fixed3 fits; not a training-seed interval')


@torch.inference_mode()
def evaluate(cohort,device):
    frozen=frozen_models();entries=base.old_entries('val') if cohort=='old_val' else read(OUT/'features/new_reserve/index.json')['cases']
    records=base.load_records(entries);require(len(records)==(172 if cohort=='old_val' else 300),'Wrong evaluation count')
    vectors={};methods={};details={r['sample_id']:dict(sample_id=r['sample_id'],source=r['source'],group=r['group'],source_group=r['source_group'],methods={}) for r in records}
    status('evaluating_'+cohort,models_completed=0,models_total=9,queries=len(records))
    for name,e in frozen['models'].items():
        require(sha(e['path'])==e['sha256'],'Frozen model drift');saved=torch.load(e['path'],weights_only=True,map_location='cpu')
        model=RetentionModel().to(device);model.load_state_dict(saved['state_dict']);model.eval();metrics={k:[] for k in ('top1','top4','union_top1')}
        for r in records:
            p=OUT/f'predictions/{cohort}/{name}/{key(r["sample_id"])}.npz'
            if p.exists():
                with np.load(p,allow_pickle=False) as z:
                    require(str(z['checkpoint_sha256'])==e['sha256'] and str(z['base_sha256'])==r['sha256'],'Cached prediction drift');scores=z['scores'].copy()
            else:
                if r['n']:
                    inputs,_=tensors([r],saved['mean'].numpy(),saved['std'].numpy(),'aligned',device);scores=model(**inputs).float().cpu().numpy()
                else:scores=np.empty(0,np.float32)
                base.npz(p,scores=scores,checkpoint_sha256=np.asarray(e['sha256']),base_sha256=np.asarray(r['sha256']))
            require(scores.shape==(r['n'],) and np.isfinite(scores).all(),'Bad scores')
            pool=np.sort(r['baseline']);order=pool[np.argsort(-scores[pool],kind='stable')];pick=int(order[0]) if len(order) else None;u=int(scores.argmax()) if len(scores) else None
            v=dict(point=r['points'][pick].tolist() if pick is not None else None,top1=bool(pick is not None and r['labels'][pick]),top4=bool(r['labels'][order[:4]].any()),union_top1=bool(u is not None and r['labels'][u]))
            details[r['sample_id']]['methods'][name]=v
            for k in metrics:metrics[k].append(v[k])
        vectors[name]=metrics['top1'];methods[name]=dict(top1_hits=sum(metrics['top1']),top4_coverage=sum(metrics['top4']),union_top1_hits=sum(metrics['union_top1']),accuracy=sum(metrics['top1'])/len(records))
        status('evaluating_'+cohort,models_completed=len(methods),models_total=9,last_model=name,queries=len(records));del model
    comparisons={};aggregate={}
    for arm in ('old_matched','old8'):
        for s in (1,2,3):
            a,b=f'expanded_s{s}',f'{arm}_s{s}';value=paired(vectors[a],vectors[b],entries,bootstrap=True)
            if cohort=='new_reserve':value['source_group_ci95']=paired(vectors[a],vectors[b],[dict(e,group=e['source_group']) for e in entries],bootstrap=True)['ci95']
            comparisons[a+' vs '+b]=value
        av=[vectors[f'expanded_s{s}'] for s in (1,2,3)];bv=[vectors[f'{arm}_s{s}'] for s in (1,2,3)]
        aggregate['expanded vs '+arm]=mean_seed_comparison(av,bv,entries)
        if cohort=='new_reserve':aggregate['expanded vs '+arm]['source_group_ci95']=mean_seed_comparison(av,bv,entries,'source_group')['ci95']
    by_source={source:dict(n=sum(r['source']==source for r in records),hits={name:sum(v for v,r in zip(values,records) if r['source']==source) for name,values in vectors.items()}) for source in sorted({r['source'] for r in records})}
    for name,v in methods.items():v['source_macro_accuracy']=float(np.mean([s['hits'][name]/s['n'] for s in by_source.values()]))
    result=dict(complete=True,cohort=cohort,n=len(records),methods=methods,comparisons=comparisons,seed_mean_comparisons=aggregate,by_source=by_source,
        rgb_groups=len({r['group'] for r in records}),source_groups={s:len({r['source_group'] for r in records if r['source']==s}) for s in by_source},
        original_k16_coverage=sum(bool(r['labels'][r['baseline']].any()) for r in records),union_coverage=sum(bool(r['labels'].any()) for r in records),empty_queries=sum(not r['n'] for r in records),
        scope='Development-only static localization, no final independent test or actual clicks; all3 seeds reported')
    write(OUT/f'evaluation/{cohort}/details.json',list(details.values()));write(OUT/f'evaluation/{cohort}/results.json',result);render();del records;gc.collect()


def complete():
    register();base.legacy_hashes()
    for filename, expected in read(base.V13/'frozen.json')['files'].items():require(sha(filename)==expected,'V13 sealed artifact drift')
    paths=[OUT/'protocol.json',OUT/'models/frozen.json',OUT/'features/new_train/index.json',OUT/'features/new_reserve/index.json']
    paths += [OUT/f'evaluation/{c}/{f}.json' for c in ('old_val','new_reserve') for f in ('results','details')]
    require(all(read(p).get('complete',True) for p in paths),'Incomplete stage')
    write(OUT/'completion.json',dict(complete=True,completed_at_utc=datetime.now(timezone.utc).isoformat(),artifact_sha256={str(p):sha(p) for p in paths},
        source_identity=source_identity(),legacy_unchanged=True,new_models=7,evaluated_models=9,independent_final_test=False))
    status('complete',new_models=7,evaluated_models=9,old_val_queries=172,new_development_queries=300)


def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=('register','extract-train','train','freeze','extract-reserve','evaluate','all'),default='all');p.add_argument('--device',default='cuda:0');p.add_argument('--limit',type=int);args=p.parse_args()
    os.chdir(ROOT);torch.set_num_threads(4);register()
    try:
        if args.stage in ('extract-train','all'):extract('new_train',args.limit)
        if args.limit is not None:return
        if args.stage in ('train','all'):train_all(args.device)
        if args.stage in ('freeze','all'):freeze()
        if args.stage in ('extract-reserve','all'):extract('new_reserve')
        if args.stage in ('evaluate','all'):
            for c in ('old_val','new_reserve'):evaluate(c,args.device)
            complete()
    except BaseException as exc:
        prior=read(OUT/'status.json') if (OUT/'status.json').exists() else {}
        write(OUT/f'failures/{datetime.now().strftime("%Y%m%d_%H%M%S")}.json',dict(error=repr(exc),traceback=traceback.format_exc(),prior=prior))
        status('failed',error=repr(exc),previous_phase=prior.get('phase'),resume_command='.venv/python.exe -B -X utf8 -u scripts/run_quality_readout_v15_3k.py --stage all');raise


if __name__=='__main__':main()
