"""Small, fixed-budget data scaling pilot using authenticated existing caches.

New development reserve and historical benchmark test predictions are not read.
The old873 baseline uses the archived8-epoch checkpoints: exactly856 updates.
"""
from __future__ import annotations
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('PYTHONDONTWRITEBYTECODE', '1')
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import html
import json
from pathlib import Path
import shutil
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))
import numpy as np
import torch
import run_quality_readout_v15 as base
from clickagent.evidence_retention import RetentionModel,retention_loss,require
from run_evidence_retention_v14 import tensors,paired

OUT=ROOT/'runs/quality_readout_v15_pilot'
SEEDS=(20261311,20261312)
SOURCES=('GroundCUA','ShowUI-web','AMEX')
LEVELS=(15,30)
STEPS=856
read,write,sha,key,digest=base.read,base.write,base.sha,base.key,base.digest


def source_identity():
    original=read(base.OUT/'protocol.json')['source_identity']
    for name,expected in original.items():require(sha(ROOT/name)==expected,'Original source drift: '+name)
    extra=(Path(__file__),ROOT/'tests/test_quality_readout_v15_pilot.py')
    return dict(original,**{str(p.relative_to(ROOT)):sha(p) for p in extra})


def select_subset(records,per_source):
    """Source-balanced nested prefixes, independent of all model outputs/cache availability."""
    selected=[]
    for source in SOURCES:
        cohort=sorted((r for r in records if r['dataset']==source),key=lambda r:key(r['sample_id']))
        require(len(cohort)>=per_source,'Not enough source cases')
        selected.extend(cohort[:per_source])
    require(len({r['sample_id'] for r in selected})==len(selected),'Repeated pilot query')
    return selected


def render():
    state=read(OUT/'status.json') if (OUT/'status.json').exists() else {}
    label={'registered':'协议已固定','training':'正在训练小规模扩容评分器','models_frozen':'全部模型已冻结',
        'evaluating':'正在开发验证集上评估','complete':'小规模实验已完成','failed':'运行遇到错误'}.get(state.get('phase'),'准备中')
    text=['<h1>V15 小规模数据扩容实验</h1>',f'<p><strong>{label}</strong></p>',
        '<p>原873条 → 加45条（每来源15条）→ 加90条（每来源30条）。全部使用相同22,050参数模型、两个种子、固定856步。复用已有特征，不等待全量提取。</p>',
        '<p>当前全量采集已让位于本小试。旧172条验证有历史复用，本轮为开发探索；新预留集不参与。短指令、悬停指令及无候选样本不按模型表现筛掉。</p>']
    if state.get('phase')=='training':
        step=state.get('step',0);text.append(f'<h2>{html.escape(state.get("model",""))}：{step} / {STEPS} 步</h2><progress max="{STEPS}" value="{step}" style="width:100%"></progress>')
    text.append('<pre>'+html.escape(json.dumps(state,ensure_ascii=False,indent=2))+'</pre>')
    p=OUT/'evaluation/results.json'
    if p.exists():
        result=read(p)
        text.append('<h2>同一原K16候选中的最终Top1命中</h2><table><tr><th>训练数据</th><th>S1 /172</th><th>S2 /172</th></tr>')
        for name,title in [('old','原873条'),('add45','873+45=918条'),('add90','873+90=963条')]:
            text.append(f'<tr><td>{title}</td><td>{result["methods"][name+"_s1"]["top1_hits"]}</td><td>{result["methods"][name+"_s2"]["top1_hits"]}</td></tr>')
        text.append('</table><p>样本较少；本轮不能替代全量实验或独立泛化测试。配对区间仅包含固定模型下的验证集抽样不确定性。</p>')
        text.append('<h2>分来源结果</h2><table><tr><th>验证来源</th><th>题数</th><th>原数据 S1/S2</th><th>+45 S1/S2</th><th>+90 S1/S2</th></tr>')
        for source,v in result['by_source'].items():
            cells=[f'{v["hits"][n+"_s1"]} / {v["hits"][n+"_s2"]}' for n in ('old','add45','add90')]
            text.append(f'<tr><td>{html.escape(source)}</td><td>{v["n"]}</td>'+''.join('<td>'+x+'</td>' for x in cells)+'</tr>')
        text.append('</table>')
    text.append('<p><a href="protocol.json">固定协议</a> · <a href="evaluation/results.json">完整结果</a> · <a href="status.json">技术状态</a> · <a href="../quality_readout_v15/progress.html">原全量实验进度</a></p>')
    page='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta http-equiv="refresh" content="20"><title>ClickAgent 小规模扩容实验</title><style>body{max-width:1050px;margin:36px auto;padding:0 24px;background:#f4f7fb;color:#24364b;font:16px/1.7 Segoe UI,Microsoft YaHei,sans-serif}table{width:100%;border-collapse:collapse;background:white}td,th{padding:10px;text-align:left;border-bottom:1px solid #ddd}pre{white-space:pre-wrap;background:white;padding:16px}a{color:#176e9e}</style><body>'+'\n'.join(text)+'</body></html>'
    p=OUT/'report.html';tmp=p.with_suffix('.html.tmp');tmp.write_text(page,encoding='utf-8');base.replace(tmp,p)


def status(phase,**extra):
    state=dict(phase=phase,complete=phase=='complete',updated_at_utc=datetime.now(timezone.utc).isoformat(),**extra)
    write(OUT/'status.json',state);render();print(json.dumps(state,ensure_ascii=False),flush=True)


def register():
    p=OUT/'protocol.json'
    if p.exists():
        v=read(p);require(v['source_identity']==source_identity(),'Pilot source drift')
        for name,expected in v['input_sha256'].items():require(sha(name)==expected,'Pilot input drift: '+name)
        return
    receipt=read(base.OUT/'analysis/data_suitability/qc1_candidate/preparation.json')
    manifest=Path(receipt['candidate_manifests']['train.jsonl']['path'])
    require(sha(manifest)==receipt['candidate_manifests']['train.jsonl']['sha256'],'Clean candidate changed')
    cohort=base.rows(manifest);old=base.old_entries('train')
    require(len(old)==873,'Wrong old train count')
    valid_old=sum(e['n']>0 for e in old);require(8*((valid_old+7)//8)==STEPS,'Archived8 epochs not matched')
    all_selected=select_subset(cohort,30);entries=[]
    for c in all_selected:
        ep=base.OUT/f'features/new_train/cases/{key(c["sample_id"])}.json'
        require(ep.exists(),'Required fixed-prefix feature missing: '+c['sample_id'])
        e=read(ep);require(e['case_identity']==digest(c),'Pilot cached case changed');entries.append(e)
    old_val_meta=[e for e in read(base.OLD/'base/index.json')['cases'] if e['split']=='val']
    require(not {e['group'] for e in old+entries}&{e['group'] for e in old_val_meta},'Pilot train/val RGB overlap')
    by_id={e['sample_id']:e for e in entries}
    write(OUT/'data/old_train_index.json',old)
    inputs={str(manifest):sha(manifest),str(base.OUT/'protocol.json'):sha(base.OUT/'protocol.json'),
        str(base.OLD/'base/index.json'):sha(base.OLD/'base/index.json'),str(base.OLD/'states/index.json'):sha(base.OLD/'states/index.json')}
    for count in LEVELS:
        chosen=select_subset(cohort,count);es=[by_id[c['sample_id']] for c in chosen]
        path=OUT/f'data/add{count*3}_index.json';write(path,es);inputs[str(path)]=sha(path)
        path=OUT/f'data/add{count*3}.jsonl';path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(''.join(json.dumps(c,ensure_ascii=False)+'\n' for c in chosen),encoding='utf-8');inputs[str(path)]=sha(path)
    inputs[str(OUT/'data/old_train_index.json')]=sha(OUT/'data/old_train_index.json')
    for s in (1,2):
        for suffix in ('.json','.pt'):
            rp=base.OLD/f'models/aligned_s{s}{suffix}';inputs[str(rp)]=sha(rp)
    spec=dict(created_at_utc=datetime.now(timezone.utc).isoformat(),source_identity=source_identity(),input_sha256=inputs,
        question='小规模增加清理后的新来源样本，在相同优化步数下是否改善现有定位评分器？',
        selection='First15/30 cases per source under original V15 sample-ID SHA order, after the preexisting placeholder-only QC filter; nested source-balanced prefixes. No model outcome, candidate reachability or cache completion filter.',
        regimes={'old':873,'add45':918,'add90':963},new_sources=list(SOURCES),seeds=list(SEEDS),steps=STEPS,
        baseline='Archived V14 aligned8-epoch S1/S2.851 nonempty fitting queries,107 updates/epoch,856 updates total. Reconstruct train-only normalization and initialization before reusing. The V15 old13400-step models are not same-budget primary controls.',
        training='Identical22,050-parameter RetentionModel, aligned states, batch8, AdamW2e-4 wd.02, clip1, dropout0, original listwise/null+.25 balancedBCE; same initialized seed, deterministic epoch shuffle. Each expanded run stops at856 updates. No validation-based checkpoint, seed or sample-size selection.',
        normalization='Per-regime candidate-base mean/std fitted on that regime train only; archived old scaler reproduced exactly. State features unchanged.',
        empty_queries='Fixed selected query cohorts include no-name/no-candidate failures. Skip zero-candidate samples in fitting exactly as legacy; evaluation denominator retains every case.',
        primary='Forced Top1 within unchanged original actual K16 pool, stable canonical union-index tie order, all172 old validation cases.',
        secondary=['Top4 coverage within original K16','Direct whole-union Top1 separately','By-source hits and source macro rate'],
        comparisons='Eachseed:add45 vsold,add90 vsold,add90 vsadd45. Report every arm/seed; no winner selection.',
        uncertainty='Paired source-stratified RGB-cluster bootstrap2000, fixed trained-model conditional intervals.',
        evaluation='Freeze all4 new checkpoints and2 old references before any pilot validation prediction. No new reserve labels/features/predictions or historical192 benchmark predictions are consumed.',
        limits=['Historically reused development validation, not independent test.',
            'Only90 added queries; a negative result cannot rule out benefit at12500 new queries.',
            'This balanced small-prefix mixture differs from the full-data source proportions.',
            'Placeholder cleanup is complete for selected rows; the separate90-query semantic audit remains pending. Do not call these fully manually certified labels.',
            'Only static target localization; no behavior selection or real clicks.'],
        user_priority='Full acquisition temporarily interrupted with all caches/old controls preserved so this pilot can finish first.')
    archive=OUT/'source_archive'
    for name in spec['source_identity']:
        source=ROOT/name;dest=archive/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,dest)
    write(p,spec);status('registered',regimes=spec['regimes'],steps=STEPS,seeds=list(SEEDS))


def load_training():
    old=base.load_records(read(OUT/'data/old_train_index.json'))
    new90=base.load_records(read(OUT/'data/add90_index.json'))
    lookup={r['sample_id']:r for r in new90}
    regimes={}
    for count in LEVELS:
        ids=[r['sample_id'] for r in read(OUT/f'data/add{count*3}_index.json')]
        regimes[f'add{count*3}']=old+[lookup[sid] for sid in ids]
    valid=[r for r in old if r['n']];x=np.concatenate([r['base'] for r in valid]);mean=x.mean(0);std=x.std(0);std[std<1e-5]=1
    checks={}
    for s in (1,2):
        receipt=read(base.OLD/f'models/aligned_s{s}.json');saved=torch.load(receipt['checkpoint_path'],weights_only=True,map_location='cpu')
        require(np.array_equal(mean,saved['mean'].numpy()) and np.array_equal(std,saved['std'].numpy()),'Old normalization replay differs')
        require(sha(receipt['checkpoint_path'])==receipt['checkpoint_sha256'],'Old reference drift')
        checks[f's{s}']=dict(checkpoint_sha256=receipt['checkpoint_sha256'],steps=STEPS,train_only_scaler_byte_identical=True)
    write(OUT/'checks/baseline_reuse.json',dict(complete=True,old_queries=len(old),old_nonempty=len(valid),references=checks))
    return regimes


def train_one(name,records,seed,device):
    tag=f'{name}_s{SEEDS.index(seed)+1}';path=OUT/f'models/{tag}.json'
    if path.exists():
        v=read(path);require(v['complete'] and sha(v['checkpoint_path'])==v['checkpoint_sha256'],'Finished pilot checkpoint drift');return
    valid=[r for r in records if r['n']];x=np.concatenate([r['base'] for r in valid]);mean=x.mean(0);std=x.std(0);std[std<1e-5]=1
    torch.manual_seed(seed);np.random.seed(seed);model=RetentionModel().to(device)
    initial=digest({k:hashlib.sha256(v.detach().cpu().numpy().tobytes()).hexdigest() for k,v in model.state_dict().items()})
    require(initial==read(base.OLD/f'models/aligned_s{SEEDS.index(seed)+1}.json')['initial_state_sha256'],'Pilot init differs')
    optimizer=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=.02);history=[];epoch=1;position=0;step=0;elapsed=0.
    rp=OUT/f'models/{tag}.resume.pt'
    if rp.exists():
        saved=torch.load(rp,weights_only=True,map_location='cpu');require(saved['protocol_sha256']==sha(OUT/'protocol.json'),'Pilot resume protocol drift')
        model.load_state_dict(saved['state_dict']);optimizer.load_state_dict(saved['optimizer']);step=saved['step'];epoch=saved['epoch'];position=saved['position'];history=saved['history'];elapsed=saved['elapsed']
    start=time.perf_counter();losses=[];model.train()
    status('training',model=tag,step=step,total_steps=STEPS,queries=len(records),fitting_queries=len(valid),empty_skipped=len(records)-len(valid))
    while step<STEPS:
        ix,epoch,position=base.next_batch(len(valid),seed,epoch,position);batch=[valid[i] for i in ix]
        inputs,lengths=tensors(batch,mean,std,'aligned',device);optimizer.zero_grad(set_to_none=True)
        scores=model(**inputs);loss=torch.stack([retention_loss(s,torch.as_tensor(r['labels'],device=device),model.null) for s,r in zip(scores.split(lengths),batch)]).mean()
        require(bool(torch.isfinite(loss)),'Nonfinite pilot loss');loss.backward()
        require(all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters()),'Nonfinite pilot gradient')
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();step+=1;losses.append(float(loss.detach()))
        if step%200==0 or step==STEPS:
            h=dict(step=step,epoch=epoch,position=position,train_loss=float(np.mean(losses)),elapsed_seconds=elapsed+time.perf_counter()-start);history.append(h);losses=[]
            rp.parent.mkdir(parents=True,exist_ok=True);tmp=rp.with_suffix('.pt.tmp')
            torch.save(dict(state_dict=model.state_dict(),optimizer=optimizer.state_dict(),step=step,epoch=epoch,position=position,history=history,elapsed=h['elapsed_seconds'],protocol_sha256=sha(OUT/'protocol.json')),tmp);base.replace(tmp,rp)
            status('training',model=tag,total_steps=STEPS,queries=len(records),fitting_queries=len(valid),**h)
    ckpt=OUT/f'models/{tag}.pt';tmp=ckpt.with_suffix('.pt.tmp')
    torch.save(dict(state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},mean=torch.tensor(mean),std=torch.tensor(std),variant='aligned',seed=seed),tmp);base.replace(tmp,ckpt)
    write(path,dict(complete=True,name=tag,seed=seed,steps=STEPS,queries=len(records),fitting_queries=len(valid),empty_skipped=len(records)-len(valid),
        parameter_count=sum(p.numel() for p in model.parameters()),initial_state_sha256=initial,checkpoint_path=str(ckpt),checkpoint_sha256=sha(ckpt),
        protocol_sha256=sha(OUT/'protocol.json'),history=history,validation_used_for_fitting_or_selection=False))
    del model,optimizer
    if torch.cuda.is_available():torch.cuda.empty_cache()


def freeze():
    models={}
    for s in (1,2):
        e=read(base.OLD/f'models/aligned_s{s}.json');models[f'old_s{s}']=dict(path=e['checkpoint_path'],sha256=e['checkpoint_sha256'],steps=STEPS,queries=873)
        for count in LEVELS:
            name=f'add{count*3}_s{s}';e=read(OUT/f'models/{name}.json')
            require(e['complete'] and e['steps']==STEPS and sha(e['checkpoint_path'])==e['checkpoint_sha256'],'Pilot model not complete')
            models[name]=dict(path=e['checkpoint_path'],sha256=e['checkpoint_sha256'],steps=STEPS,queries=e['queries'])
    value=dict(complete=True,models=models,source_identity=source_identity(),protocol_sha256=sha(OUT/'protocol.json'))
    path=OUT/'models/frozen.json'
    if path.exists():require(read(path)==value,'Pilot seal drift')
    else:write(path,value)
    status('models_frozen',models=6,new_models=4)


@torch.inference_mode()
def evaluate(device):
    frozen=read(OUT/'models/frozen.json');require(frozen['complete'],'Pilot models not frozen')
    records=base.load_records(base.old_entries('val'));require(len(records)==172,'Wrong validation count')
    vectors={};methods={};details={r['sample_id']:dict(sample_id=r['sample_id'],source=r['source'],group=r['group'],methods={}) for r in records};replay={}
    for name,m in frozen['models'].items():
        require(sha(m['path'])==m['sha256'],'Frozen pilot model drift');saved=torch.load(m['path'],weights_only=True,map_location='cpu')
        model=RetentionModel().to(device);model.load_state_dict(saved['state_dict']);model.eval();metrics={k:[] for k in ('top1','top4','union_top1')};maximum=0.
        for r in records:
            if r['n']:
                inputs,_=tensors([r],saved['mean'].numpy(),saved['std'].numpy(),'aligned',device);scores=model(**inputs).float().cpu().numpy()
            else:scores=np.empty(0,np.float32)
            if name.startswith('old_'):
                pp=base.OLD/f'predictions/aligned_{name.split("_")[-1]}/val/{key(r["sample_id"])}.npz'
                with np.load(pp,allow_pickle=False) as z:expected=z['scores'].copy()
                require(np.allclose(scores,expected,atol=1e-5,rtol=1e-5),'Archived baseline inference differs')
                maximum=max(maximum,float(np.abs(scores-expected).max(initial=0)))
            pool=np.sort(r['baseline']);order=pool[np.argsort(-scores[pool],kind='stable')]
            pick=int(order[0]) if len(order) else None;u=int(scores.argmax()) if len(scores) else None
            value=dict(point=r['points'][pick].tolist() if pick is not None else None,top1=bool(pick is not None and r['labels'][pick]),
                top4=bool(r['labels'][order[:4]].any()),union_top1=bool(u is not None and r['labels'][u]))
            details[r['sample_id']]['methods'][name]=value
            for k in metrics:metrics[k].append(value[k])
            base.npz(OUT/f'predictions/{name}/{key(r["sample_id"])}.npz',scores=scores,checkpoint_sha256=np.asarray(m['sha256']),base_sha256=np.asarray(r['sha256']))
        vectors[name]=metrics['top1'];methods[name]=dict(top1_hits=sum(metrics['top1']),top4_coverage=sum(metrics['top4']),union_top1_hits=sum(metrics['union_top1']),n=172,accuracy=sum(metrics['top1'])/172)
        if name.startswith('old_'):replay[name]=dict(max_abs_score_difference=maximum,atol=1e-5,rtol=1e-5)
        status('evaluating',models_completed=len(methods),models_total=6,last_model=name,hits=methods[name]['top1_hits']);del model
    comparisons={}
    for s in (1,2):
        for a,b in [(f'add45_s{s}',f'old_s{s}'),(f'add90_s{s}',f'old_s{s}'),(f'add90_s{s}',f'add45_s{s}')]:
            comparisons[a+' vs '+b]=paired(vectors[a],vectors[b],records,bootstrap=True)
    by_source={source:dict(n=sum(r['source']==source for r in records),hits={n:sum(h for h,r in zip(v,records) if r['source']==source) for n,v in vectors.items()}) for source in sorted({r['source'] for r in records})}
    for name in methods:methods[name]['source_macro_accuracy']=float(np.mean([v['hits'][name]/v['n'] for v in by_source.values()]))
    result=dict(complete=True,n=172,methods=methods,comparisons=comparisons,by_source=by_source,
        original_k16_coverage=sum(bool(r['labels'][r['baseline']].any()) for r in records),empty_validation_queries=sum(not r['n'] for r in records),
        scope='Small balanced-data pilot on historically reused development validation; no new reserve, independent benchmark or actual behavior execution. A negative pilot cannot rule out gains from larger/new-distribution training.')
    write(OUT/'evaluation/details.json',list(details.values()));write(OUT/'evaluation/results.json',result);write(OUT/'checks/baseline_inference_replay.json',dict(complete=True,models=replay))
    register()
    paths=[OUT/'protocol.json',OUT/'models/frozen.json',OUT/'evaluation/results.json',OUT/'evaluation/details.json',OUT/'checks/baseline_reuse.json',OUT/'checks/baseline_inference_replay.json']
    write(OUT/'completion.json',dict(complete=True,completed_at_utc=datetime.now(timezone.utc).isoformat(),artifact_sha256={str(p):sha(p) for p in paths},
        original_full_data_or_sources_modified=False,new_reserve_consumed=False,source_identity=source_identity()))
    status('complete',methods=methods,comparisons=comparisons)


def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=('register','all','evaluate'),default='all');p.add_argument('--device',default='cuda:0');args=p.parse_args()
    os.chdir(ROOT);torch.set_num_threads(4);register()
    try:
        if args.stage=='register':return
        if args.stage=='all':
            regimes=load_training()
            for name in ('add45','add90'):
                for seed in SEEDS:train_one(name,regimes[name],seed,args.device)
            freeze()
        evaluate(args.device)
    except BaseException as exc:
        write(OUT/'failure.json',dict(error=repr(exc),traceback=traceback.format_exc()))
        status('failed',error=repr(exc));raise


if __name__=='__main__':main()
