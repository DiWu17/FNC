"""Sequential V14 development experiments; V13 is strictly read-only.

Stage 1: rank/score calibration over fixed coordinates, matched actual count.
Stage 2: aligned own-token states vs the same model with no extra states and
spatially permuted states. Stage 3 is registered only after stage 2 completes.
No historical test prediction is an experimental input.
"""
from __future__ import annotations
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('PYTHONDONTWRITEBYTECODE', '1')
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import html
import json
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
sys.path.insert(0, str(ROOT/'scripts'))
import numpy as np
import torch
from clickagent.evidence_retention import (BASE_DIM, STATE_DIM, RetentionModel,
    build_base, state_features, permute_evidence, select_indices, retention_loss, require)
from clickagent.public_evaluation import point_in_geometry

LEGACY = ROOT/'runs/multiregion_v13'
OUT = ROOT/'runs/evidence_retention_v14'
SEEDS = (20261311, 20261312)
SOURCES = (Path(__file__), ROOT/'src/clickagent/evidence_retention.py')
VERIFIED_STATES = {}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def write(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    temp.replace(path)


def save_npz(path, **arrays):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp')
    with temp.open('wb') as f:np.savez(f,**arrays)
    temp.replace(path)


def key(sid):return hashlib.sha256(sid.encode()).hexdigest()[:24]


def cases(split):
    require(split in ('train','val'),'Only development train/val are authorized here')
    return [json.loads(l) for l in (LEGACY/f'data/{split}.jsonl').read_text(encoding='utf-8').splitlines() if l.strip()]


def source_identity():return {str(p.relative_to(ROOT)):sha(p) for p in SOURCES}


def completion_hashes():
    record=read(LEGACY/'completion.json'); found={}
    for base,group in ((LEGACY,'artifact_sha256'),(ROOT,'docs_sha256')):
        for name,expected in record[group].items():
            path=base/name;actual=sha(path)
            require(actual==expected,f'Sealed completion file changed: {path}')
            found[str(path)]=actual
    return found


def snapshot(stage):
    folder=OUT/stage/'source_archive';folder.mkdir(parents=True,exist_ok=True)
    values=source_identity()
    for path in SOURCES:shutil.copyfile(path,folder/path.name)
    write(folder/'identity.json',values)
    return values


def register(stage, specification):
    path=OUT/stage/'protocol.json'
    if path.exists():
        old=read(path);require(old['specification']==specification,'Protocol drift')
        return old
    value=dict(created_at_utc=datetime.now(timezone.utc).isoformat(),specification=specification,
        source_identity=snapshot(stage),legacy_completion_hashes=completion_hashes(),
        historical_test_used_for_policy_selection=False)
    write(path,value)
    return value


def initialize():
    OUT.mkdir(parents=True,exist_ok=True)
    path=OUT/'decision.json'
    if not path.exists():
        write(path,dict(date='2026-11-03',user_direction='先收敛最终贡献，按顺序设计并执行三个实验。基础思路与已有工作有交集本身不否定novelty。',
            contribution_hypothesis='利用冻结模型生成目标名称时的内部空间证据，指导多区域搜索，并在有限候选预算内保留正确定位机会。',
            scope='本轮先隔离候选保留：裁剪、全部已解码坐标、上游提议器固定。没有实现新的搜索框，不把保留实验归因为搜索收益。',
            questions=['已有正确点能否在相同实际候选数下保住？','额外内部状态及其空间对应是否有独立价值？','可达性收益能否经固定内部头和GUI验证器转成实际点命中？'],
            sequencing='实验1完成后登记实验2；实验2完成后登记实验3。负结果同样继续完成对照，不按验证结果换主种子。',
            data='沿用873训练、172验证；均为开发实验，验证集有历史复用，不称新独立泛化证据。V13测试/补充不用于新策略选择。',
            legacy='V13所有原件只读；不调用原写回审计CLI；新代码及输出独立。',
            auxiliary='Top4专门损失、来源CV保留为背景，不追加为本轮主创新。'))
    write(OUT/'work_state.json',dict(phase='initialized',complete=False,decision_sha256=sha(path)))
    render_report()


def render_report():
    decision=read(OUT/'decision.json')
    chunks=['<h1>ClickAgent：贡献收敛与候选保留实验</h1>',
        '<p>2026-11-03 · 开发实验 · V13 原件只读</p>',
        '<p><strong>'+html.escape(decision['contribution_hypothesis'])+'</strong></p>',
        '<p>'+html.escape(decision['scope'])+'</p>',
        '<p>'+html.escape(decision['data'])+'</p>',
        '<p>共享基础思路不自动否定 novelty。本轮以新增机制及其可归因收益收敛贡献；未来若没有相应收益，不把假设写成结论。</p>']
    labels=['问题一：相同实际点数下保留已有正确点','问题二：额外内部状态及空间对应的作用','问题三：固定下游选择的最终命中']
    for i,label in enumerate(labels,1):
        stage=f'experiment{i}';path=OUT/stage/'results.json'
        chunks.append('<h2>'+label+'</h2>')
        chunks.append('<p>协议：<a href="'+stage+'/protocol.json">protocol.json</a> · '+('已完成' if path.exists() else '尚未完成')+'</p>')
        if not path.exists():continue
        result=read(path)
        if i<3:
            chunks.append('<table><tr><th>方法</th><th>训练可达 / 873</th><th>验证可达 / 172</th><th>验证相对原配额：赢 / 输</th></tr>')
            for name in result['val']['matched_coverage']:
                comp=result['val']['comparisons'].get(name)
                wl='—' if comp is None else f"{comp['wins']} / {comp['losses']}"
                chunks.append(f"<tr><td>{html.escape(name)}</td><td>{result['train']['matched_coverage'][name]}</td><td>{result['val']['matched_coverage'][name]}</td><td>{wl}</td></tr>")
            chunks.append('</table><p>可达性采用 GT 事后几何判定，不是最终命中。主比较逐题采用原 K16 的实际点数；cap16 补位另列于 JSON。训练为拟合诊断，验证有历史复用。</p>')
            if i==2:
                chunks.append('<p>rank 不增读原始状态；aligned 保留全部目标名称 token、96 头及实际点上的注意力/语义对应；permuted 在每个视图内打乱点与证据对应。三个分支参数量、优化、种子相同。</p>')
        else:
            chunks.append('<pre>'+html.escape(json.dumps(result.get('summary',result),ensure_ascii=False,indent=2))+'</pre>')
    chunks.append('<h2>预先确定的解释边界</h2><p>不根据 V13 测试选新策略。没有重新生成候选坐标、没有改变裁剪或 Qwen 读取预算。先研究保留机制；新的搜索策略需要后续独立对照。Top4 损失与来源 CV 不作为本轮新增主贡献。</p>')
    css="body{max-width:1050px;margin:40px auto;padding:0 24px;background:#f8fafc;color:#23354b;font:16px/1.7 'Segoe UI','Microsoft YaHei',sans-serif}h1,h2{line-height:1.4}h2{margin-top:38px}table{width:100%;border-collapse:collapse;background:white}th,td{padding:10px 14px;border-bottom:1px solid #dce4ed;text-align:left}th{background:#e8eff6}pre{white-space:pre-wrap}a{color:#176e9e}"
    (OUT/'report.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>ClickAgent V14 贡献实验</title><style>'+css+'</style><body>'+'\n'.join(chunks)+'</body></html>',encoding='utf-8')


def prepare_base():
    path=OUT/'base/index.json'
    if path.exists():
        index=read(path);require(index['complete'],'Incomplete base index');return
    audit=read(LEGACY/'final_audit.json')
    expected={os.path.normcase(os.path.abspath(p)):v for p,v in audit['input_sha256'].items()}
    entries=[];started=time.perf_counter()
    for split,n in (('train',873),('val',172)):
        cohort=cases(split);require(len(cohort)==n,'Wrong fixed split size')
        require(sha(LEGACY/f'data/{split}.jsonl')==expected[os.path.normcase(os.path.abspath(LEGACY/f'data/{split}.jsonl'))],'Manifest SHA changed')
        for i,case in enumerate(cohort,1):
            sid=key(case['sample_id']);pp=LEGACY/f'pipeline/{split}/{sid}.json'
            ph=sha(pp);require(ph==expected[os.path.normcase(os.path.abspath(pp))],'Pipeline SHA changed')
            pipe=read(pp)
            require(pipe['complete'] and pipe['sample_id']==case['sample_id'],'Pipeline identity mismatch')
            require(not pipe['ground_truth_in_model_inputs'],'Legacy GT input')
            base=build_base(pipe,case['image_size'])
            points=base['points']
            labels=np.asarray([point_in_geometry((p/np.asarray(case['image_size'])).tolist(),case['target_geometry']) for p in points],np.float32)
            bp=OUT/f'base/{split}/{sid}.npz';lp=OUT/f'labels/{split}/{sid}.npy'
            save_npz(bp,sample_id=np.asarray(case['sample_id']),pipeline_sha256=np.asarray(ph),**base)
            lp.parent.mkdir(parents=True,exist_ok=True);np.save(lp,labels,allow_pickle=False)
            entries.append(dict(sample_id=case['sample_id'],split=split,source=case['dataset'],
                group=case['pixel_sha256'],image_fold_group=case.get('image_fold_group'),
                path=str(bp),sha256=sha(bp),labels_path=str(lp),labels_sha256=sha(lp),
                n=len(points),matched_count=len(base['baseline']),pipeline_path=str(pp),pipeline_sha256=ph))
            if i%100==0 or i==n:print(json.dumps(dict(stage='prepare_base',split=split,i=i,n=n)),flush=True)
    write(path,dict(complete=True,cases=entries,source_identity=source_identity(),
        input_geometry_is_supervision_only=True,inference_feature_keys=['base','active','points','baseline'],
        elapsed_seconds=time.perf_counter()-started))


def load_data(variant, split=None):
    index=read(OUT/'base/index.json');require(index['complete'],'Need base')
    records=[]
    state_index=read(OUT/'states/index.json') if variant!='rank' else None
    states={e['sample_id']:e for e in state_index['cases']} if state_index else {}
    for entry in index['cases']:
        if split is not None and entry['split'] != split:
            continue
        require(sha(entry['path'])==entry['sha256'] and sha(entry['labels_path'])==entry['labels_sha256'],'Base/label changed')
        with np.load(entry['path'],allow_pickle=False) as z:
            record={k:z[k].copy() for k in ('points','base','active','baseline')}
        record.update(entry)
        record['labels']=np.load(entry['labels_path'],allow_pickle=False)
        record['state']=record['mask']=None
        if variant!='rank':
            e=states[entry['sample_id']]
            for field,sha_field in (('path','sha256'),('mask_path','mask_sha256')):
                p=Path(e[field]);stat=p.stat();stamp=(stat.st_size,stat.st_mtime_ns)
                if str(p) not in VERIFIED_STATES:
                    require(sha(p)==e[sha_field],'Extracted state changed')
                    VERIFIED_STATES[str(p)]=stamp
                require(VERIFIED_STATES[str(p)]==stamp,'State changed during experiment')
            record['state']=np.load(e['path'],mmap_mode='r',allow_pickle=False)
            record['mask']=np.load(e['mask_path'],mmap_mode='r',allow_pickle=False)
        records.append(record)
    return records


def tensors(records, mean, std, variant, device):
    lengths=[r['n'] for r in records];n=sum(lengths)
    base=np.concatenate([(r['base']-mean)/std for r in records])
    args=dict(base=torch.as_tensor(base,device=device),active=torch.as_tensor(np.concatenate([r['active'] for r in records]),device=device))
    if variant!='rank':
        tmax=max(r['state'].shape[2] for r in records)
        state=np.zeros((n,4,tmax,STATE_DIM),np.float16);mask=np.zeros((n,4,tmax),bool)
        start=0
        for r in records:
            v=r['state'];m=r['mask'];t=v.shape[2]
            if variant=='permuted':
                rng=np.random.default_rng(int(digest([r['sample_id'],'spatial_permutation'])[:8],16))
                v=permute_evidence(v,m,rng)
            state[start:start+r['n'],:,:t]=v;mask[start:start+r['n'],:,:t]=m;start+=r['n']
        args.update(state=torch.as_tensor(state,device=device),mask=torch.as_tensor(mask,device=device))
    return args,lengths


def train(variant, seed, device):
    name=f'{variant}_s{SEEDS.index(seed)+1}'
    receipt=OUT/f'models/{name}.json'
    if receipt.exists():
        value=read(receipt);require(value['complete'] and sha(value['checkpoint_path'])==value['checkpoint_sha256'],'Model receipt mismatch');return
    stage='experiment1' if variant=='rank' else 'experiment2'
    protocol=read(OUT/stage/'protocol.json');identity=source_identity()
    require(identity==protocol['source_identity'],'Source changed after registration')
    records=[r for r in load_data(variant,split='train') if r['n']]
    x=np.concatenate([r['base'] for r in records])
    mean=x.mean(0);std=x.std(0);std[std<1e-5]=1
    torch.manual_seed(seed);np.random.seed(seed)
    model=RetentionModel().to(device)
    initial=digest({k:hashlib.sha256(v.detach().cpu().numpy().tobytes()).hexdigest() for k,v in model.state_dict().items()})
    optimizer=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=.02)
    history=[];started=time.perf_counter()
    for epoch in range(1,9):
        order=np.random.default_rng(seed+epoch*101).permutation(len(records));model.train();losses=[]
        for start in range(0,len(order),8):
            batch=[records[i] for i in order[start:start+8]]
            inputs,lengths=tensors(batch,mean,std,variant,device)
            optimizer.zero_grad(set_to_none=True);scores=model(**inputs)
            chunks=scores.split(lengths)
            loss=torch.stack([retention_loss(s,torch.as_tensor(r['labels'],device=device),model.null) for s,r in zip(chunks,batch)]).mean()
            require(bool(torch.isfinite(loss)),'Nonfinite loss')
            loss.backward();require(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()),'Nonfinite gradient')
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();losses.append(float(loss.detach()))
        history.append(dict(epoch=epoch,train_loss=float(np.mean(losses)),elapsed_seconds=time.perf_counter()-started))
        write(OUT/f'models/{name}_progress.json',dict(name=name,history=history,validation_used_for_fitting=False))
        print(json.dumps(dict(stage='training',name=name,**history[-1])),flush=True)
    model.eval();ckpt=OUT/f'models/{name}.pt';ckpt.parent.mkdir(parents=True,exist_ok=True)
    torch.save(dict(state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},mean=torch.tensor(mean),std=torch.tensor(std),variant=variant,seed=seed),ckpt)
    require(identity==source_identity(),'Source changed during training')
    write(receipt,dict(complete=True,name=name,variant=variant,seed=seed,epochs=8,
        checkpoint_path=str(ckpt),checkpoint_sha256=sha(ckpt),initial_state_sha256=initial,
        parameter_count=sum(p.numel() for p in model.parameters()),protocol_sha256=sha(OUT/stage/'protocol.json'),
        base_index_sha256=sha(OUT/'base/index.json'),source_identity=identity,history=history,
        validation_used_for_epoch_or_hyperparameter_selection=False))


@torch.inference_mode()
def predict(variant,seed,device):
    name=f'{variant}_s{SEEDS.index(seed)+1}';receipt=read(OUT/f'models/{name}.json')
    records=load_data(variant);saved=torch.load(receipt['checkpoint_path'],weights_only=True,map_location='cpu')
    model=RetentionModel().to(device);model.load_state_dict(saved['state_dict']);model.eval()
    mean,std=saved['mean'].numpy(),saved['std'].numpy()
    for i,r in enumerate(records,1):
        dest=OUT/f'predictions/{name}/{r["split"]}/{key(r["sample_id"])}.npz'
        if dest.exists():
            with np.load(dest,allow_pickle=False) as z:require(str(z['checkpoint_sha256'])==receipt['checkpoint_sha256'],'Stale prediction')
            continue
        if r['n']:
            inputs,_=tensors([r],mean,std,variant,device);scores=model(**inputs).cpu().numpy()
        else:scores=np.empty(0,np.float32)
        save_npz(dest,sample_id=np.asarray(r['sample_id']),scores=scores,
            matched=select_indices(scores,r['matched_count']),cap16=select_indices(scores,min(16,r['n'])),
            checkpoint_sha256=np.asarray(receipt['checkpoint_sha256']),base_sha256=np.asarray(r['sha256']))
        if i%100==0 or i==len(records):print(json.dumps(dict(stage='predict',name=name,i=i,n=len(records))),flush=True)


def paired(a,b,entries,bootstrap=True):
    a=np.asarray(a,bool);b=np.asarray(b,bool)
    value=dict(n=len(a),method_hits=int(a.sum()),reference_hits=int(b.sum()),wins=int((a&~b).sum()),losses=int((b&~a).sum()),delta=float(a.mean()-b.mean()))
    if bootstrap:
        strata=defaultdict(lambda:defaultdict(list))
        for i,e in enumerate(entries):strata[e['source']][e['group']].append(i)
        groups=[list(s.values()) for s in strata.values()];rng=np.random.default_rng(20261401)
        values=[]
        for _ in range(2000):
            ix=[i for cluster in groups for j in rng.integers(0,len(cluster),len(cluster)) for i in cluster[j]]
            values.append(float((a[ix].astype(float)-b[ix]).mean()))
        value['ci95']=np.quantile(values,[.025,.975]).tolist()
        value['uncertainty']='Source-stratified RGB-cluster bootstrap, fixed trained models; development validation reused historically'
    return value


def retention_results(stage,names):
    index=read(OUT/'base/index.json')['cases'];result={};all_rows=[]
    for split in ('train','val'):
        entries=[e for e in index if e['split']==split];hits={n:[] for n in ['baseline']+names};caps={n:[] for n in names}
        for e in entries:
            y=np.load(e['labels_path'],allow_pickle=False)
            with np.load(e['path'],allow_pickle=False) as z:baseline=z['baseline'].tolist();points=z['points'].copy()
            pools=dict(baseline=baseline);hits['baseline'].append(bool(y[baseline].any()))
            for name in names:
                p=OUT/f'predictions/{name}/{split}/{key(e["sample_id"])}.npz'
                with np.load(p,allow_pickle=False) as z:
                    ix=z['matched'];cap=z['cap16']
                    require(len(ix)==len(baseline),'Actual count not matched')
                    require(len(set(ix.tolist()))==len(ix),'Repeated candidate')
                    pools[name]=ix.tolist();hits[name].append(bool(y[ix].any()));caps[name].append(bool(y[cap].any()))
            all_rows.append(dict(sample_id=e['sample_id'],split=split,source=e['source'],group=e['group'],
                matched_count=len(baseline),points=points.tolist(),pools=pools,coverage={n:v[-1] for n,v in hits.items()}))
        row=dict(n=len(entries),rgb_groups=len({e['group'] for e in entries}),
            matched_coverage={n:sum(v) for n,v in hits.items()},cap16_secondary_coverage={n:sum(v) for n,v in caps.items()},
            comparisons={n:paired(hits[n],hits['baseline'],entries,bootstrap=split=='val') for n in names},
            by_source={s:dict(n=sum(e['source']==s for e in entries),coverage={n:sum(x for x,e in zip(v,entries) if e['source']==s) for n,v in hits.items()}) for s in sorted({e['source'] for e in entries})})
        if stage=='experiment2':
            row['aligned_vs_rank']={f's{i}':paired(hits[f'aligned_s{i}'],hits[f'rank_s{i}'],entries,bootstrap=split=='val') for i in (1,2)}
            row['aligned_vs_permuted']={f's{i}':paired(hits[f'aligned_s{i}'],hits[f'permuted_s{i}'],entries,bootstrap=split=='val') for i in (1,2)}
        result[split]=row
    result['complete']=True;result['scope']='Development train/val only. Train scores are in-sample fit diagnostics; validation is historically reused, not fresh independent confirmation.'
    write(OUT/stage/'results.json',result)
    (OUT/stage/'pools.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in all_rows),encoding='utf-8')
    write(OUT/stage/'completion.json',dict(complete=True,protocol_sha256=sha(OUT/stage/'protocol.json'),
        results_sha256=sha(OUT/stage/'results.json'),pools_sha256=sha(OUT/stage/'pools.jsonl'),
        models={n:sha(OUT/f'models/{n}.pt') for n in names},legacy_completion_hashes=completion_hashes()))
    render_report()
    print(json.dumps(dict(stage=stage,complete=True,validation=result['val'])),flush=True)
    return result


def experiment1(device):
    spec=dict(question='相同裁剪、全部解码坐标、逐题相同实际候选数下，已有正确点能否更好保留？',
        universe='All archived c3 decoded integer coordinates, exact dedup, global ranked points then local round-robin; no GT point injection.',
        primary_budget='m_i = len(sealed c3 K16 pool) per case; all arms select exactly m_i. Canonical coordinate-universe order passed downstream.',
        secondary_budget='cap16 = min(16, decoded universe size), kept separate from matched-count primary.',
        controls='Sealed fixed half-global half-local quota vs shared learned rank/score calibration across all views.',
        features='32 public proposal features: per-view source membership, log-rank, reciprocal rank, raw score, within-view z-score, rank fraction, log-count; point xy, occurrence and available-view count.',
        internal_limitation='Proposer scores/ranks already derive from internal states. This baseline reads no EXTRA raw states; it is not an attention-free pipeline.',
        model='Identical architecture reserved for stage2; extra 294 state channels zero in rank arm; exact parameter count recorded from model.',
        optimization=dict(epochs=8,batch_queries=8,lr=.0002,weight_decay=.02,gradient_clip=1.,dropout=0,
            loss='multi-positive listwise/null + .25 query-balanced point BCE',seeds=list(SEEDS)),
        selection='Fixed epoch8, no validation-selected epoch/hyperparameter/seed. S1 primary, S2 robustness.',
        outputs='Coverage, paired gains/losses, source breakdown, conditional fixed-model CI on development validation; no final click claim in stage1.',
        order='Register -> prepare inputs -> train both seeds -> freeze both -> predict train/val -> analyze.')
    register('experiment1',spec);prepare_base()
    for seed in SEEDS:train('rank',seed,device)
    for seed in SEEDS:predict('rank',seed,device)
    result=retention_results('experiment1',['rank_s1','rank_s2'])
    write(OUT/'work_state.json',dict(phase='experiment1_complete',complete=False,validation=result['val']))


def prepare_states():
    ip=OUT/'states/index.json'
    if ip.exists():require(read(ip)['complete'],'Incomplete state extraction');return
    entries=[];raw_ledger={};started=time.perf_counter()
    cohort={c['sample_id']:c for split in ('train','val') for c in cases(split)}
    expected=read(LEGACY/'final_audit.json')['input_sha256']
    expected={os.path.normcase(os.path.abspath(p)):v for p,v in expected.items()}
    for i,e in enumerate(read(OUT/'base/index.json')['cases'],1):
        case=cohort[e['sample_id']]
        pipe=read(e['pipeline_path'])
        require(sha(e['pipeline_path'])==e['pipeline_sha256'],'Pipeline changed')
        with np.load(e['path'],allow_pickle=False) as z:points=z['points'].copy()
        views=[dict(feature_path=pipe['global_feature_path'],feature_sha256=pipe['global_feature_sha256'],transform=None)]+pipe['local_views']
        features=[];masks=[]
        for view in views:
            path=Path(view['feature_path']);canonical=os.path.normcase(os.path.abspath(path))
            if canonical not in raw_ledger:
                actual=sha(path);require(actual==view['feature_sha256']==expected[canonical],'Raw state SHA changed')
                raw_ledger[canonical]=actual
            with np.load(path,allow_pickle=False) as z:
                meta=json.loads(str(z['metadata']));require(meta['ground_truth_in_model_input'] is False,'GT raw input')
                require(meta['own_token_forward'] is True,'Need actual own-token states')
                arrays={k:z[k] for k in ('target_attention','target_hidden','image_hidden')}
            f,m=state_features(arrays,meta,points,case['image_size'],view['transform'])
            features.append(f);masks.append(m)
        tmax=max(1,*(f.shape[1] for f in features));state=np.zeros((len(points),4,tmax,STATE_DIM),np.float16);mask=np.zeros((len(points),4,tmax),bool)
        for j,(f,m) in enumerate(zip(features,masks)):
            t=f.shape[1];state[:,j,:t]=f;mask[:,j,:t]=m[:,None]
        path=OUT/f'states/{e["split"]}/{key(e["sample_id"])}.npy';mp=path.with_suffix('.mask.npy')
        path.parent.mkdir(parents=True,exist_ok=True);np.save(path,state,allow_pickle=False);np.save(mp,mask,allow_pickle=False)
        entries.append(dict(sample_id=e['sample_id'],path=str(path),mask_path=str(mp),sha256=sha(path),mask_sha256=sha(mp),shape=list(state.shape),base_sha256=e['sha256']))
        if i%50==0 or i==1045:print(json.dumps(dict(stage='extract_extra_states',i=i,n=1045,seconds=time.perf_counter()-started)),flush=True)
    write(ip,dict(complete=True,cases=entries,raw_sha256=raw_ledger,raw_forward_calls=0,
        all_name_tokens_preserved=True,all_96_head_identities_preserved=True,geometry_not_accepted_by_builder=True,
        feature_schema='96 log relative center +96 relative to head peak +96 log local mean +3 query/image cosines +3 query lognorms',elapsed_seconds=time.perf_counter()-started))


def experiment2(device):
    require(read(OUT/'experiment1/completion.json')['complete'],'Experiment1 must complete first')
    spec=dict(question='相同模型、训练、点集、实际候选数下，额外目标token内部证据及其正确空间对应是否提高保留？',
        preceding_experiment_sha256=sha(OUT/'experiment1/results.json'),
        architecture='Exact same initialized RetentionModel as rank-only, same number of parameters, fixed8epochs/two seeds/optimization/schedule.',
        arms=['rank: no extra raw channels, from experiment1','aligned: target token/head evidence at each actual candidate point','permuted: same features shuffled among candidate points independently within each view, preserving view availability and all token identities'],
        features='294 per-token channels; every actual target token and96 heads retained until learned token/candidate interaction, then token max/logmeanexp and view max/mean.',
        limits=['Tests this explicit attention + query/image similarity representation, not all possible internal states.',
            'Shared proposer rankings already contain internal information; isolates EXTRA evidence and spatial alignment.',
            'Spatial permutation is a matched training-and-evaluation negative control, not a proof of a unique causal feature.'],
        loader_note='Stage2 filters split before opening records. Stage1 shared loader had opened and then excluded val records; exact train-only scaler reconstruction is saved in checks/fit_split.json. This changes no training example, model, loss, initialization or normalization.',
        selection='No additional epoch/hyperparameter selection; S1 primary, S2 robustness. Do not swap primary arm for a validation winner.',
        order='Register after experiment1 -> authenticate raw archives -> extract -> train all four models -> freeze -> predict -> compare.')
    register('experiment2',spec);prepare_states()
    for variant in ('aligned','permuted'):
        for seed in SEEDS:train(variant,seed,device)
    for variant in ('aligned','permuted'):
        for seed in SEEDS:predict(variant,seed,device)
    result=retention_results('experiment2',['rank_s1','rank_s2','aligned_s1','aligned_s2','permuted_s1','permuted_s2'])
    write(OUT/'work_state.json',dict(phase='experiment2_complete',complete=False,validation=result['val']))


def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=('init','experiment1','experiment2','prepare_states'),required=True)
    p.add_argument('--device',default='cuda:0');args=p.parse_args();os.chdir(ROOT);torch.set_num_threads(4)
    if args.stage=='init':initialize()
    elif args.stage=='experiment1':experiment1(args.device)
    elif args.stage=='experiment2':experiment2(args.device)
    elif args.stage=='prepare_states':prepare_states()


if __name__=='__main__':main()
