"""Fixed-data contribution ablations followed by sealed fresh confirmation.

The adopted full model is never selected/replaced from ablation/test outcomes.
Legacy training is reused through explicit output/status/input-only adapters;
its scientific source and the2973-query training cohort remain byte-identical.
"""
from __future__ import annotations
import os
os.environ.setdefault('PYTHONDONTWRITEBYTECODE','1')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse
from contextlib import contextmanager
from datetime import datetime,timezone
import gc
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
from PIL import Image
import torch
import run_quality_readout_v15_3k as prior
from clickagent.contribution_ablation import VARIANTS,transform_inputs
from clickagent.evidence_retention import RetentionModel,require
from clickagent.public_diagnostics import parse_offline_lax_click

OUT=ROOT/'runs/contribution_v16';MAIN=ROOT/'runs/quality_readout_v15_3k'
SEEDS=(20261311,20261312,20261313);STEPS=2976
read,write,sha,key,digest=prior.read,prior.write,prior.sha,prior.key,prior.digest
ORIGINAL_TENSORS=prior.tensors
TITLES={'aligned':'完整空间证据（已采用）','no_evidence':'去掉内部证据值','spatial_shuffle':'打乱证据与候选的空间对应',
    'attention_only':'仅保留注意力通道','hidden_only':'仅保留隐藏状态通道','token_mean':'先平均名称token再编码'}
NEW_SOURCES=('scripts/run_contribution_v16.py','src/clickagent/contribution_ablation.py','tests/test_contribution_v16.py',
    'scripts/prepare_contribution_v16.py','scripts/prepare_semantic_head_data.py','scripts/prepare_candidate_scorer.py',
    'src/clickagent/public_diagnostics.py','scripts/run_gui_verifier.py','src/clickagent/point_aligned_inference.py')


def identity():
    old=read(MAIN/'protocol.json')['source_identity']
    for name,expected in old.items():require(sha(ROOT/name)==expected,'Adopted scientific source drift: '+name)
    return dict(old,**{name:sha(ROOT/name) for name in NEW_SOURCES})


def render():
    s=read(OUT/'status.json') if (OUT/'status.json').exists() else {}
    phase=s.get('phase','registered')
    labels={'registered':'方案与数据已固定','loading_train':'读取并校验既有2973条训练特征','training':'正在训练消融评分器',
        'models_frozen':'18个检查点已冻结','extracting_independent':'独立300条：自然名称特征及原生坐标对照',
        'gui_reference':'独立300条：外部GUI验证器对照','complete':'贡献消融与独立确认已完成','failed':'运行遇到错误'}
    label=labels.get(phase,'正在评估 '+phase.removeprefix('evaluating_'))
    body=['<h1>固定2973条与22,050参数：贡献消融和独立确认</h1>',
        '<p>完整模型保持为已采用版本；另外5组消融×3种子，共15次新训练。每次2976步，候选、上游、损失和优化设置一致。</p>',
        '<p>先完成训练与开发消融，再自动处理首次使用的独立300张截图：ScreenSpot-Pro150＋VenusBench-GD150。图片与所有历史及导入数据严格排重。</p>',
        f'<div class="card"><h2>{html.escape(label)}</h2>']
    if phase=='training':body.append(f'<p>{html.escape(s.get("model",""))}：{s.get("step",0)} / {s.get("total_steps",STEPS)}步</p>')
    if phase in ('extracting_independent','gui_reference'):
        body.append(f'<p><b>{s.get("completed",0)} / {s.get("total",300)}</b></p><progress value="{s.get("completed",0)}" max="300"></progress>')
    if phase.startswith('evaluating_'):body.append(f'<p>已评分 {s.get("models_completed",0)} / {s.get("models_total",18)} 个模型。</p>')
    if phase=='failed':body.append('<p>'+html.escape(s.get('error',''))+'</p>')
    body.append('<p class="muted">更新时间（UTC）：'+html.escape(s.get('updated_at_utc',''))+'；30秒自动刷新。</p></div>')
    for cohort,title in [('old_val','历史旧开发验证172条'),('new_development','已使用的新来源开发验证300条'),('independent','独立确认300张新截图')]:
        p=OUT/f'evaluation/{cohort}/results.json'
        if not p.exists():continue
        r=read(p);body.append(f'<h2>{title}</h2><table><tr><th>输入证据</th><th>S1命中</th><th>S2命中</th><th>S3命中</th><th>平均准确率</th></tr>')
        for v in VARIANTS:
            hits=[r['methods'][f'{v}_s{i}']['top1_hits'] for i in (1,2,3)]
            body.append('<tr><td>'+TITLES[v]+'</td>'+''.join(f'<td>{h}/{r["n"]}</td>' for h in hits)+f'<td>{100*np.mean(hits)/r["n"]:.2f}%</td></tr>')
        body.append('</table>')
        if cohort=='independent':
            body.append('<h3>固定外部及上游参考</h3><ul>')
            for name in ('proposal_first','qwen_native_strict','qwen_native_completed_lax','gui_verifier_k16'):
                body.append(f'<li>{name}：{r["methods"][name]["top1_hits"]}/{r["n"]}</li>')
            body.append('</ul>')
    body.extend(['<div class="card"><b>结论边界</b><p>消融只隔离固定上游和固定候选条件下，内部证据对评分的增量价值。候选提议器本身已使用Qwen内部状态。</p><p>独立队列是两个基准的首次使用子集，不能作为完整官方榜单分数。截图排重不保证应用或大模型预训练未见；所有结果只涉及静态定位。</p><p>原生Qwen只看全图，当前方法使用全图和局部；GUI参考使用额外外部训练过的2B验证器。分别报告配置，不宣称计算预算相等。</p></div>',
        '<p><a href="protocol.json">固定实验方案</a> · <a href="data/independent_selection.json">独立数据隔离审计</a> · <a href="status.json">技术状态</a> · <a href="logs/events.jsonl">运行日志</a> · <a href="../quality_readout_v15_3k/report.html">已采用版本</a></p>'])
    page='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta http-equiv="refresh" content="30"><title>ClickAgent 贡献消融与独立确认</title><style>body{max-width:1160px;margin:32px auto;padding:0 24px;font:16px/1.7 Segoe UI,Microsoft YaHei,sans-serif;background:#f3f6fb;color:#24364b}.card{padding:20px;background:white;border-radius:12px;margin:20px 0}.muted{color:#63758b;font-size:14px}table{width:100%;background:white;border-collapse:collapse}td,th{padding:9px;border-bottom:1px solid #ddd;text-align:left}progress{width:100%;height:22px}a{color:#176e9e}</style><body>'+'\n'.join(body)+'</body></html>'
    p=OUT/'report.html';temp=p.with_suffix('.html.tmp');temp.write_text(page,encoding='utf-8');prior.base.replace(temp,p)


def status(phase,**extra):
    s=dict(phase=phase,complete=phase=='complete',pid=os.getpid(),updated_at_utc=datetime.now(timezone.utc).isoformat(),**extra)
    write(OUT/'status.json',s);p=OUT/'logs/events.jsonl';p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('a',encoding='utf-8') as stream:stream.write(json.dumps(s,ensure_ascii=False)+'\n')
    render();print(json.dumps(s,ensure_ascii=False),flush=True)


def register():
    path=OUT/'protocol.json'
    if path.exists():
        p=read(path);require(p['source_identity']==identity(),'Contribution source drift')
        for name,h in p['input_sha256'].items():require(sha(name)==h,'Frozen contribution input drift: '+name)
        return
    c=read(MAIN/'completion.json');require(c['complete'],'Adopted run not complete')
    for name,h in c['artifact_sha256'].items():require(sha(name)==h,'Adopted artifact drift')
    selection=read(OUT/'data/independent_selection.json');require(selection['complete'] and selection['selected_queries']==300,'Confirmation cohort not prepared')
    for name,h in selection['input_sha256'].items():require(sha(name)==h,'Selection input drift')
    require(sha(OUT/'data/independent_test.jsonl')==selection['manifest_sha256'],'Confirmation membership drift')
    inputs={str(MAIN/'completion.json'):sha(MAIN/'completion.json'),**c['artifact_sha256'],
        str(OUT/'data/independent_selection.json'):sha(OUT/'data/independent_selection.json'),str(OUT/'data/independent_test.jsonl'):sha(OUT/'data/independent_test.jsonl')}
    inputs.update(selection['input_sha256'])
    for p in [MAIN/'data/old_train_index.json',MAIN/'data/new_train.jsonl',ROOT/'research/decisions/20261104_adopt_quality_readout_v15_3k.json',
        ROOT/'models/GUI-Actor-Verifier-2B/clickagent_download_verified.json',prior.base.V12/'runtime_identity.json',prior.base.V12/'proposers/full.pt']:
        inputs[str(p)]=sha(p)
    seal=read(MAIN/'models/frozen.json')
    for s in (1,2,3):
        p=MAIN/f'models/expanded_s{s}.json';inputs[str(p)]=sha(p)
        e=seal['models'][f'expanded_s{s}'];require(sha(e['path'])==e['sha256'],'Adopted checkpoint drift');inputs[e['path']]=e['sha256']
    spec=dict(created_at_utc=datetime.now(timezone.utc).isoformat(),source_identity=identity(),input_sha256=inputs,
        mainline='Already adopted2973-query aligned scorer; remains main model regardless of ablation/development/test outcomes.',
        train_queries=2973,steps=STEPS,seeds=list(SEEDS),parameters=22050,
        training='Exact unmodified V15_3k train_one, with only registered output/status/tensors adapters. Same query ordering, batch8, AdamW2e-4 wd.02, clip1, listwise/null+.25balancedBCE, train-only scaler and fixed final2976 updates. Zero-candidate fit records skipped identically; full evaluation denominators retained.',
        variants={'aligned':'Authenticated3 adopted checkpoints reused; no refitting.',
            'no_evidence':'Zero all294 state values, preserve original masks/token presence, base metadata, active views and shapes. Proposer itself still uses internal states; tests incremental scoring information.',
            'spatial_shuffle':'Frozen V14 deterministic within-valid-view candidate permutation keyed by sampleID; keep masks, base, head/token channels and marginal values. Train/evaluate consistently shuffled.',
            'attention_only':'Keep288 attention channels; zero6 hidden query/image cosine and query-norm channels.',
            'hidden_only':'Zero288 attention channels; keep3 query/image cosines plus3 query-norm channels.',
            'token_mean':'Masked arithmetic mean over real name tokens before state encoder, broadcast to original real-token positions; masks unchanged. No additional backbone generation.'},
        physical_capacity='Identical RetentionModel class and22,050 stored parameters in every arm; ablations change information/effective use of capacity, not architecture. No input-only mask/coordinate/candidate-count or crop-budget changes.',
        checkpoint_metadata='New ablation checkpoint tensors are unchanged after fit; feature_ablation metadata is added before freeze so the inference recipe is explicit.',
        controls='All15 new fits and3 full references freeze before any new ablation development prediction or independent image forward. No best seed, checkpoint, arm or threshold selection.',
        independent_test=dict(n=300,sources={'ScreenSpot-Pro':150,'VenusBench-GD':150},selection_sha256=sha(OUT/'data/independent_selection.json'),
            exclusion='All discovered project manifest histories and entire imported train/reserve image pools; exactRGB and dHash256<=4 exclusion including within selected cohort. One instruction per fresh screenshot; labels only used for geometry scoring after public-only extraction.',
            claim='Fresh project-level image-isolated confirmation subset; not full official benchmarks, not guaranteed unseen software or pretraining data.'),
        development='Old172 and already consumed new300 remain development only. Results are reported without changing the adopted full method or independent protocol.',
        primary='Top1 integer-point hit within same actual originalK16, stable canonical union-index tie order. Independent full-vs-no_evidence, full-vs-spatial_shuffle, and full-vs-native_completed_lax are3 fixed primary comparisons.',
        secondary=['Other3 feature ablations','All3 seed outcomes and mean','Same-pool frozen GUI-Actor-Verifier-2B','Quota-first upstream point','Native strict parser separately','OriginalK16/union coverage and Top4; no GUI Top4 execution claim'],
        references={'qwen_native':'Same frozen Qwen, original public coordinate prompt, full-image262144 pixels, greedy32tokens, no hooks or shared trajectory; strict and completed-lax parsing both reported, malformed/WAIT count as misses.',
            'gui_verifier':'Frozen official2B verifier, original prompt and marked crop, same canonical originalK16 points, fixed maxbatch4; True/False log-odds and forcedbest point. External trained reference, no teacher/fitting use.',
            'proposal_first':'First point in original unmodified quota pool, no learned quality scoring.'},
        uncertainty='Source-stratified RGB-cluster paired bootstrap2000 over seed-averaged predictions. Eachseed discordant wins/losses additionally. Independent3-primary family uses98.333% Bonferroni intervals;95% marginal intervals separately. App/source-group cluster sensitivity intervals also reported. Conditional on fixed3 fits, not training-seed population uncertainty.',
        workflow=['train15 ablation models','freeze18 models and source/input identities','both fixed development evaluations','independent name-feature/native-coordinate extraction','same-pool GUI reference','independent metrics for all18 checkpoints and4 fixed references','integrity audit and complete'],
        limitations=['Upstream candidate generator already uses name states; readout ablations do not isolate their whole-pipeline necessity.',
            'Token-mean is a feature pooling control, not evidence that target-name time is uniquely optimal over all decoding stages.',
            'No app-level or pretrained-data novelty claim from screenshot exclusion; selected public datasets may have lineage overlap.',
            'Full method uses more views than global-native baseline; GUI adds an external2B model. No matched-FLOPs or speed superiority claim.',
            'No actual click execution, calibrated abstention or action-type learning.',
            'Separate90-query semantic audit remains pending; do not describe every training label as manually certified.'])
    for name in spec['source_identity']:
        dest=OUT/'source_archive'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,dest)
    write(path,spec);status('registered',new_fits=15,checkpoints=18,train_queries=2973,independent_queries=300)


def build_inputs(records,mean,std,variant,device):
    args,lengths=ORIGINAL_TENSORS(records,mean,std,'aligned',device)
    return transform_inputs(args,lengths,[r['sample_id'] for r in records],variant),lengths


@contextmanager
def adapters(variant='aligned'):
    saved=(prior.OUT,prior.status,prior.tensors)
    prior.OUT=OUT;prior.status=status
    prior.tensors=lambda records,mean,std,ignored,device:build_inputs(records,mean,std,variant,device)
    try:yield
    finally:prior.OUT,prior.status,prior.tensors=saved


def state_digest(weights):
    return digest({k:hashlib.sha256(v.detach().cpu().numpy().tobytes()).hexdigest() for k,v in weights.items()})


def train_all(device):
    status('loading_train',queries=2973)
    entries=read(MAIN/'data/old_train_index.json')+read(MAIN/'features/new_train/index.json')['cases']
    records=prior.base.load_records(entries);require(len(records)==2973,'Training cohort changed')
    for variant in VARIANTS[1:]:
        for seed in SEEDS:
            with adapters(variant):prior.train_one(variant,records,seed,STEPS,device)
            tag=f'{variant}_s{SEEDS.index(seed)+1}';path=OUT/f'models/{tag}.json';r=read(path)
            full=read(MAIN/f'models/expanded_s{SEEDS.index(seed)+1}.json')
            require(r['initial_state_sha256']==full['initial_state_sha256'],'Ablation initialization differs')
            checkpoint=torch.load(r['checkpoint_path'],map_location='cpu',weights_only=True)
            full_checkpoint=torch.load(full['checkpoint_path'],map_location='cpu',weights_only=True)
            require(torch.equal(checkpoint['mean'],full_checkpoint['mean']) and torch.equal(checkpoint['std'],full_checkpoint['std']),'Ablation scaler differs')
            require(r['parameter_count']==22050 and r['queries']==2973,'Capacity or data changed')
            if r.get('feature_ablation')!=variant:
                before=state_digest(checkpoint['state_dict']);checkpoint.update(variant=variant,feature_ablation=variant)
                p=Path(r['checkpoint_path']);temp=p.with_suffix('.pt.recipe.tmp');torch.save(checkpoint,temp);prior.base.replace(temp,p)
                require(state_digest(torch.load(p,map_location='cpu',weights_only=True)['state_dict'])==before,'Metadata update changed model tensors')
                r.update(feature_ablation=variant,model_state_sha256=before,checkpoint_sha256=sha(p));write(path,r)
    del records;gc.collect()


def freeze():
    models={};original=read(MAIN/'models/frozen.json')['models']
    for variant in VARIANTS:
        for s in (1,2,3):
            name=f'{variant}_s{s}'
            if variant=='aligned':e=original[f'expanded_s{s}'];path=e['path'];h=e['sha256']
            else:
                e=read(OUT/f'models/{name}.json');require(e['complete'] and e['feature_ablation']==variant and e['steps']==STEPS,'Ablation not complete');path=e['checkpoint_path'];h=e['checkpoint_sha256']
            require(sha(path)==h,'Checkpoint bytes changed');models[name]=dict(path=path,sha256=h,variant=variant,seed=SEEDS[s-1])
    value=dict(complete=True,models=models,protocol_sha256=sha(OUT/'protocol.json'),source_identity=identity(),independent_features_at_initial_freeze=0)
    p=OUT/'models/frozen.json'
    if p.exists():require(read(p)==value,'Frozen contribution models drift')
    else:
        require(not list((OUT/'independent').rglob('*.npz')) and not list((OUT/'native').glob('*.json')),'Fresh prediction occurred before model freeze')
        write(p,value)
    status('models_frozen',models=18,new_fits=15)


def require_frozen():
    p=OUT/'models/frozen.json';require(p.exists(),'Freeze18 models before independent inputs')
    v=read(p);require(v.get('complete') and len(v['models'])==18,'Incomplete contribution model freeze')
    require(v['source_identity']==identity() and v['protocol_sha256']==sha(OUT/'protocol.json'),'Frozen inference recipe drift')
    for e in v['models'].values():require(sha(e['path'])==e['sha256'],'Frozen model file changed')
    return v


def extract_independent():
    require_frozen();manifest=OUT/'data/independent_test.jsonl';cases=prior.base.rows(manifest);ip=OUT/'independent/index.json'
    if ip.exists():
        r=read(ip);require(r['complete'] and r['manifest_sha256']==sha(manifest) and r['models_frozen_sha256']==sha(OUT/'models/frozen.json'),'Independent cache drift');return
    entries=[];started=time.perf_counter();extractor=None;status('extracting_independent',completed=0,total=300)
    with adapters():
        for i,c in enumerate(cases,1):
            dest=OUT/f'independent/cases/{key(c["sample_id"])}.json';npth=OUT/f'native/{key(c["sample_id"])}.json'
            if dest.exists():
                e=read(dest);prior.authenticate_entry(e,c);require(sha(npth)==e['native_sha256'],'Native cache changed');entries.append(e)
            else:
                if extractor is None:extractor=prior.Extractor()
                public={k:c[k] for k in prior.base.PUBLIC}
                if not npth.exists():
                    require(sha(c['image'])==c['image_sha256'],'Independent image changed')
                    with Image.open(c['image']) as opened:image=opened.convert('RGB')
                    try:r=extractor.get_runtime().infer_text_action_public(image,c['instruction'],max_new_tokens=32)
                    finally:image.close()
                    require(r['hooks_used'] is False and r['shared_trajectory'] is False,'Native baseline used internal hooks')
                    complete=r['terminated_by_eos'] or r['generated_text'].endswith(('\n','\r'))
                    parsed=parse_offline_lax_click(r['generated_text']) if complete else None
                    r.update(completed_lax_point=[round(v*(size-1)/1000) for v,size in zip(parsed,c['image_size'])] if parsed else None,
                        sample_id=c['sample_id'],image_sha256=c['image_sha256'],instruction_sha256=hashlib.sha256(c['instruction'].encode()).hexdigest(),protocol_sha256=sha(OUT/'protocol.json'))
                    write(npth,r)
                else:
                    r=read(npth);require(r['image_sha256']==c['image_sha256'] and r['instruction_sha256']==hashlib.sha256(c['instruction'].encode()).hexdigest() and r['protocol_sha256']==sha(OUT/'protocol.json'),'Native cache identity drift')
                pipe,pp=extractor.case(public);b,labels,state,mask=prior.compact(c,pipe,pp)
                bp=OUT/f'independent/arrays/{key(c["sample_id"])}.npz';prior.base.npz(bp,**b)
                lp=bp.with_suffix('.labels.npy');sp=bp.with_suffix('.state.npy');mp=bp.with_suffix('.mask.npy')
                for p,a in [(lp,labels),(sp,state),(mp,mask)]:
                    temp=p.with_suffix(p.suffix+'.tmp')
                    with temp.open('wb') as stream:np.save(stream,a,allow_pickle=False)
                    prior.base.replace(temp,p)
                e=dict(sample_id=c['sample_id'],source=c['dataset'],group=c['pixel_sha256'],source_group=c['source_group'],role='independent',split='independent_test',case_identity=digest(c),n=len(b['points']),
                    path=str(bp),sha256=sha(bp),labels_path=str(lp),labels_sha256=sha(lp),state_path=str(sp),state_sha256=sha(sp),mask_path=str(mp),mask_sha256=sha(mp),pipeline_path=str(pp),pipeline_sha256=sha(pp),native_path=str(npth),native_sha256=sha(npth))
                write(dest,e);entries.append(e)
            if i%10==0 or i==300:status('extracting_independent',completed=i,total=300,elapsed_seconds=time.perf_counter()-started)
        del extractor;gc.collect()
        if torch.cuda.is_available():torch.cuda.empty_cache()
    write(ip,dict(complete=True,cases=entries,n=300,manifest_sha256=sha(manifest),models_frozen_sha256=sha(OUT/'models/frozen.json')))


def gui_reference():
    require_frozen();cases=prior.base.rows(OUT/'data/independent_test.jsonl');entries={e['sample_id']:e for e in read(OUT/'independent/index.json')['cases']}
    from run_gui_verifier import VerifierRuntime
    runtime=None;status('gui_reference',completed=0,total=300)
    for i,c in enumerate(cases,1):
        e=entries[c['sample_id']];p=OUT/f'gui_reference/{key(c["sample_id"])}.json'
        if p.exists():require(read(p)['base_sha256']==e['sha256'] and read(p)['protocol_sha256']==sha(OUT/'protocol.json'),'GUI cache drift')
        else:
            with np.load(e['path'],allow_pickle=False) as z:indices=np.sort(z['baseline']);points=z['points'][indices].tolist()
            rows=[]
            if points:
                if runtime is None:
                    runtime=VerifierRuntime();write(OUT/'gui_reference/runtime.json',runtime.metadata)
                require(sha(c['image'])==c['image_sha256'],'GUI image changed')
                with Image.open(c['image']) as opened:image=opened.convert('RGB')
                try:
                    for start in range(0,len(points),4):rows.extend(runtime.score(image,c['instruction'],points[start:start+4],'crop'))
                finally:image.close()
            write(p,dict(sample_id=c['sample_id'],base_sha256=e['sha256'],protocol_sha256=sha(OUT/'protocol.json'),scores=rows,canonical_k16_indices=indices.tolist(),ground_truth_in_model_input=False))
        if i%10==0 or i==300:status('gui_reference',completed=i,total=300)
    del runtime;gc.collect()
    if torch.cuda.is_available():torch.cuda.empty_cache()
    write(OUT/'gui_reference/completion.json',dict(complete=True,n=300,files={str(OUT/f'gui_reference/{key(c["sample_id"])}.json'):sha(OUT/f'gui_reference/{key(c["sample_id"])}.json') for c in cases}))


def bootstrap_mean(a,b,entries,group='group',primary=False):
    diff=np.asarray(a,float).mean(0)-np.asarray(b,float).mean(0);strata={}
    for i,e in enumerate(entries):strata.setdefault(e['source'],{}).setdefault(e[group],[]).append(i)
    groups=[list(v.values()) for v in strata.values()];rng=np.random.default_rng(20261604);values=[]
    for _ in range(2000):
        ix=[i for clusters in groups for j in rng.integers(0,len(clusters),len(clusters)) for i in clusters[j]];values.append(float(diff[ix].mean()))
    result=dict(delta=float(diff.mean()),ci95=np.quantile(values,[.025,.975]).tolist(),conditional_on_fixed_models=True)
    if primary:result.update(primary_family_size=3,bonferroni_ci98_333=np.quantile(values,[.05/6,1-.05/6]).tolist())
    return result


@torch.inference_mode()
def evaluate(cohort,device):
    frozen=require_frozen();path=OUT/f'evaluation/{cohort}/results.json'
    if path.exists():require(read(path)['complete'],'Incomplete evaluation');return
    entries=prior.base.old_entries('val') if cohort=='old_val' else read(MAIN/'features/new_reserve/index.json')['cases'] if cohort=='new_development' else read(OUT/'independent/index.json')['cases']
    records=prior.base.load_records(entries);vectors={};methods={};details={e['sample_id']:dict(sample_id=e['sample_id'],source=e['source'],group=e['group'],source_group=e['source_group'],methods={}) for e in entries}
    status('evaluating_'+cohort,models_completed=0,models_total=18)
    for name,e in frozen['models'].items():
        saved=torch.load(e['path'],map_location='cpu',weights_only=True);model=RetentionModel().to(device);model.load_state_dict(saved['state_dict']);model.eval();values=[];top4=[];whole=[]
        for r in records:
            p=OUT/f'predictions/{cohort}/{name}/{key(r["sample_id"])}.npz'
            reused=MAIN/f'predictions/{"old_val" if cohort=="old_val" else "new_reserve"}/expanded_s{name[-1]}/{key(r["sample_id"])}.npz'
            cache=p if p.exists() else reused if e['variant']=='aligned' and cohort!='independent' else None
            if cache is not None and cache.exists():
                with np.load(cache,allow_pickle=False) as z:
                    require(str(z['checkpoint_sha256'])==e['sha256'] and str(z['base_sha256'])==r['sha256'],'Cached inference identity drift');scores=z['scores'].copy()
            else:
                if r['n']:
                    inputs,_=build_inputs([r],saved['mean'].numpy(),saved['std'].numpy(),e['variant'],device);scores=model(**inputs).float().cpu().numpy()
                else:scores=np.empty(0,np.float32)
                prior.base.npz(p,scores=scores,checkpoint_sha256=np.asarray(e['sha256']),base_sha256=np.asarray(r['sha256']))
            require(scores.shape==(r['n'],) and np.isfinite(scores).all(),'Malformed scores')
            pool=np.sort(r['baseline']);order=pool[np.argsort(-scores[pool],kind='stable')];pick=int(order[0]) if len(order) else None;u=int(scores.argmax()) if len(scores) else None
            v=dict(point=r['points'][pick].tolist() if pick is not None else None,top1=bool(pick is not None and r['labels'][pick]),top4=bool(r['labels'][order[:4]].any()),union_top1=bool(u is not None and r['labels'][u]))
            details[r['sample_id']]['methods'][name]=v;values.append(v['top1']);top4.append(v['top4']);whole.append(v['union_top1'])
        vectors[name]=values;methods[name]=dict(top1_hits=sum(values),top4_coverage=sum(top4),union_top1_hits=sum(whole),accuracy=sum(values)/len(records));del model
        status('evaluating_'+cohort,models_completed=len(methods),models_total=18)
    if cohort=='independent':
        gui_seal=read(OUT/'gui_reference/completion.json')
        require(gui_seal['complete'] and gui_seal['n']==300 and len(gui_seal['files'])==300,'GUI reference not complete')
        for filename,expected in gui_seal['files'].items():require(sha(filename)==expected,'GUI result bytes changed')
        cases={c['sample_id']:c for c in prior.base.rows(OUT/'data/independent_test.jsonl')}
        for name in ('proposal_first','qwen_native_strict','qwen_native_completed_lax','gui_verifier_k16'):
            values=[]
            for r in records:
                c=cases[r['sample_id']];point=None
                if name=='proposal_first':point=r['points'][int(r['baseline'][0])].tolist() if len(r['baseline']) else None
                elif name.startswith('qwen_native'):
                    n=read(r['native_path']);require(sha(r['native_path'])==r['native_sha256'],'Native result drift');point=n['point'] if name.endswith('strict') else n['completed_lax_point']
                else:
                    g=read(OUT/f'gui_reference/{key(r["sample_id"])}.json');require(g['base_sha256']==r['sha256'],'GUI input drift')
                    point=g['scores'][int(np.argmax([x['log_odds'] for x in g['scores']]))]['point'] if g['scores'] else None
                hit=bool(point is not None and prior.base.point_in_geometry((np.asarray(point)/np.asarray(c['image_size'])).tolist(),c['target_geometry']))
                details[r['sample_id']]['methods'][name]=dict(point=point,top1=hit);values.append(hit)
            vectors[name]=values;methods[name]=dict(top1_hits=sum(values),accuracy=sum(values)/300)
    comparisons={}
    others=list(VARIANTS[1:])+(list(('proposal_first','qwen_native_strict','qwen_native_completed_lax','gui_verifier_k16')) if cohort=='independent' else [])
    for variant in others:
        aa=[vectors[f'aligned_s{s}'] for s in (1,2,3)];bb=[vectors[f'{variant}_s{s}'] if variant in VARIANTS else vectors[variant] for s in (1,2,3)]
        primary=cohort=='independent' and variant in ('no_evidence','spatial_shuffle','qwen_native_completed_lax')
        value=bootstrap_mean(aa,bb,entries,primary=primary);value['source_group_ci95']=bootstrap_mean(aa,bb,entries,'source_group')['ci95']
        value['per_seed']=[prior.paired(a,b,entries,bootstrap=False) for a,b in zip(aa,bb)]
        comparisons['aligned vs '+variant]=value
    by_source={source:dict(n=sum(e['source']==source for e in entries),hits={name:sum(h for h,e in zip(v,entries) if e['source']==source) for name,v in vectors.items()}) for source in sorted({e['source'] for e in entries})}
    for name,v in methods.items():v['source_macro_accuracy']=float(np.mean([s['hits'][name]/s['n'] for s in by_source.values()]))
    result=dict(complete=True,cohort=cohort,n=len(entries),methods=methods,comparisons=comparisons,by_source=by_source,
        original_k16_coverage=sum(bool(r['labels'][r['baseline']].any()) for r in records),union_coverage=sum(bool(r['labels'].any()) for r in records),empty_queries=sum(not r['n'] for r in records),
        checkpoint_seal_sha256=sha(OUT/'models/frozen.json'),independent=cohort=='independent',
        scope='Image-isolated300-query independent confirmation subset, not full benchmark or unseen-app claim' if cohort=='independent' else 'Historically consumed development evaluation; no checkpoint selection')
    write(OUT/f'evaluation/{cohort}/details.json',list(details.values()));write(path,result);render();del records;gc.collect()


def validate_completion_results(result,rows,n):
    require(isinstance(result,dict) and result.get('complete') is True and result['n']==n and isinstance(rows,list) and len(rows)==n,'Evaluation completeness/schema mismatch')
    require(len({r['sample_id'] for r in rows})==n,'Duplicate evaluation IDs')
    for row in rows:require(set(row['methods'])==set(result['methods']),'Incomplete detail models')
    for name,v in result['methods'].items():require(sum(row['methods'][name]['top1'] for row in rows)==v['top1_hits'],'Aggregate differs from details')


def complete():
    register();require_frozen()
    paths=[OUT/'protocol.json',OUT/'models/frozen.json',OUT/'data/independent_selection.json',OUT/'data/independent_test.jsonl',OUT/'independent/index.json',OUT/'gui_reference/completion.json']
    for cohort,n in [('old_val',172),('new_development',300),('independent',300)]:
        p=OUT/f'evaluation/{cohort}/results.json';d=OUT/f'evaluation/{cohort}/details.json';r=read(p);rows=read(d)
        validate_completion_results(r,rows,n)
        paths.extend((p,d))
    for filename,h in read(MAIN/'completion.json')['artifact_sha256'].items():require(sha(filename)==h,'Adopted results changed')
    write(OUT/'completion.json',dict(complete=True,completed_at_utc=datetime.now(timezone.utc).isoformat(),artifact_sha256={str(p):sha(p) for p in paths},source_identity=identity(),
        adopted_mainline_unchanged=True,independent_queries=300,independent_full_benchmark=False,new_fits=15,evaluated_checkpoints=18))
    status('complete',new_fits=15,checkpoints=18,independent_queries=300)


def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=('register','train','freeze','development','independent','evaluate','all'),default='all');p.add_argument('--device',default='cuda:0');args=p.parse_args()
    os.chdir(ROOT);torch.set_num_threads(4);register()
    try:
        if args.stage in ('train','all'):train_all(args.device)
        if args.stage in ('freeze','all'):freeze()
        if args.stage in ('development','all'):
            for cohort in ('old_val','new_development'):evaluate(cohort,args.device)
        if args.stage in ('independent','all'):extract_independent();gui_reference()
        if args.stage in ('evaluate','all'):evaluate('independent',args.device);complete()
    except BaseException as exc:
        write(OUT/f'failures/{datetime.now().strftime("%Y%m%d_%H%M%S")}.json',dict(error=repr(exc),traceback=traceback.format_exc()))
        status('failed',error=repr(exc),resume_command='.venv/python.exe -B -X utf8 -u scripts/run_contribution_v16.py --stage all');raise


if __name__=='__main__':main()
