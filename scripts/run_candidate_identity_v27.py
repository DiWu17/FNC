"""Protected supplemental candidates x cached fine appearance/competition pilot."""
import os
for k,v in {'PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4','MKL_NUM_THREADS':'4'}.items():os.environ.setdefault(k,v)
from collections import defaultdict
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil
import sys
import time
import traceback
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import numpy as np
import torch
from clickagent.candidate_identity_v27 import (ARMS,RIDGE,DIM,role_features,union_points,union_scores,
    retain,content_shuffle,probe_features,fit_residual,infer)
from clickagent.evidence_retention import RetentionModel
from run_evidence_retention_v14 import tensors
from run_visual_feedback_v23 import read,write,sha,hit,bootstrap
import run_prompt_readout_v25 as prior

OUT=ROOT/'runs/candidate_identity_v27'
V25=prior.OUT;V26=ROOT/'runs/location_selection_v26'
NAMES=prior.NAMES
SOURCES=[Path(__file__),ROOT/'src/clickagent/candidate_identity_v27.py',ROOT/'tests/test_candidate_identity_v27.py']
CONTRASTS=[('full','A'),('full','union_frozen'),('full','prior'),('full','visual'),
           ('full','pair_visual'),('full','shuffled'),('full','full_naive'),('visual','prior')]


def now():return datetime.now(timezone.utc).isoformat()
def identity():return {str(p):sha(p) for p in SOURCES}
def status(phase,**kw):
    r=dict(phase=phase,complete=phase=='complete',updated_at_utc=now(),pid=os.getpid(),**kw)
    write(OUT/'status.json',r);print(json.dumps(r,ensure_ascii=False),flush=True)
def path(c):return OUT/'features'/f'{prior.key(c["sample_id"])}.npz'


def prepare():
    assert not (OUT/'protocol.json').exists(),'Registered run already exists'
    test=read(OUT/'unit_tests.json');assert test['exit_code']==0 and test['source_sha256']==identity()
    cases=read(V25/'data/public.json');assert len(cases)==92 and len({c['image_sha256'] for c in cases})==92
    assert not {c['group'] for c in cases if c['cohort']=='train'}&{c['group'] for c in cases if c['cohort']!='train'}
    old=read(V25/'protocol.json');sealed={str(p):sha(p) for p in SOURCES}
    inputs=[V25/n for n in ['protocol.json','predictions.json','final_audit.json','cache_audit_index.json','data/public.json','data/labels.json']]
    inputs+=[V26/n for n in ['protocol.json','completion.json','final_audit.json']]
    inputs+=[ROOT/p for p in ['src/clickagent/evidence_retention.py','scripts/run_prompt_readout_v25.py',
        'src/clickagent/prompt_readout_v25.py','scripts/run_evidence_retention_v14.py','scripts/run_visual_feedback_v23.py']]
    sealed.update({str(p):sha(p) for p in inputs})
    for r in old['models'].values():assert sha(r['path'])==r['sha256'];sealed[r['path']]=r['sha256']
    spec=dict(registered_at_utc=now(),source_sha256=identity(),sealed_sha256=sealed,
        samples='V25 original-training56 RGB and reused development36 RGB(new24,old12), one query/RGB. No independent test. Freeze all learned probes before new development feature/scoring stage.',
        candidates='Exact stable union of complete A-name and B-context-name points. Coincident coordinates keep A frozen score; B-only points keep B frozen score. Same original6 heads. No GT-dependent supplement or deduplication. A points appear first.',
        retention='K16 per model. Naive ranks union frozen scores. Protected takes top8 A and top8 B-only, then fills vacancies by remaining global score. Actual candidate oracle coverage compared at equal K16. This does not guarantee retaining every positive candidate.',
        appearance='Use native unmerged frozen vision vectors(1152), normalized. For each actual name token and each of96 heads, its spatial attention weights a fine-vision coarse-cell mean template; normalize template, sample fine vector at candidate, cosine minus uniform-image prototype cosine. Compute mean/min across ALL actual role tokens before grouping consecutive4 heads within each layer(96->24). Aggregate views by validity. Role channels A target, B target, B container remain separate. No new visual encoding or VLM calls.',
        comparison='Candidate-to-nearest3 retained competitors within .12 normalized-image Euclidean distance: appearance difference, explicit relative availability. Candidate pools and features fixed across scoring ablations.',
        context='Secondary weak container context: B container appearance plus per-head candidate displacement/distance to container-attention centroid. This is NOT actual object box grounding, and NOT arbitrary left-of/between instruction execution. No fabricated relation labels.',
        probes=dict(arms=list(ARMS),feature_dim=DIM,architecture='Frozen original logits plus <=322-coefficient linear residual. Role/appearance/context subsets are fixed masks. prior calibrates origin/logits/missingness; visual adds fine appearance; pair_visual adds competitor contrast; full adds weak container context; shuffled has full capacity with within-origin/availability content permutation.',
            supervision='Training-only query/source-balanced positive-v-negative pairwise logistic loss with L2; no-positive pools yield no ranking gradient. No backbone/proposer/original head updates.',
            ridge=list(RIDGE),selection='7 leave-one-source-out folds within original training56. Select highest equal-source OOF top1, ties stronger ridge. Include zero-residual identity option; choose identity on tied OOF score. Final fit all original-training56. No development checkpoint/weight/feature selection.',
            warning='Base models can have seen original training samples and sources; train leave-source-out isolates residual fitting only, not independent whole-system generalization.'),
        primary=dict(cohort='new_development24',contrasts=CONTRASTS,
            statistics='5000 source-stratified RGB bootstrap,3 fixed-fit means;16 contrasts across2 families,99.6875% Bonferroni intervals. No independence assumption for seeds.'),
        advancement='full vs A >=3pp, corrected lower>0; full vs prior and shuffled corrected lower>0; old12 full-A>=-5pp. Development signal only, no automatic adoption. Only positive gate warrants separate independent confirmation.',
        ground_truth='Public sample fields and feature construction have no labels. Training annotations only fit residual. Development annotations only evaluate and attribute. No oracle-valued prediction is a method.',
        costs='All cached. Deployment union+features needs both A/B original views; not equal VLM budget vs A alone or MVP. All scoring ablations share union, fixed views and observed evidence.',
        user_authorization='试试吧：保留有效补充候选，测试同一区域的目标身份/局部视觉与锚点关系。本轮锚点明确限定为已有容器状态的弱上下文。',
        new_vlm_calls=0,mainline_unchanged=True,outputs='Python/JSON/TXT/NPZ/static PNG; no HTML')
    write(OUT/'protocol.json',spec);write(OUT/'registry_before.json',read(ROOT/'CURRENT_RESEARCH.json'))
    for p in SOURCES:
        dest=OUT/'source_archive'/p.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest)
    reg=read(ROOT/'CURRENT_RESEARCH.json');reg['active_experiment']=dict(run_id=OUT.name,status='running',phase='registered',
        protocol_path=str(OUT/'protocol.json'),status_path=str(OUT/'status.json'),mainline_unchanged=True)
    write(ROOT/'CURRENT_RESEARCH.json',reg)
    write(ROOT/'research/decisions/20261109_candidate_identity_v27.json',dict(registered_at_utc=now(),
        user_authorization=spec['user_authorization'],protocol_sha256=sha(OUT/'protocol.json'),mainline_unchanged=True))
    status('registered',training_rgb=56,development_rgb=36,new_vlm_calls=0)


def verify():
    spec=read(OUT/'protocol.json');assert spec['source_sha256']==identity()
    for p,h in spec['sealed_sha256'].items():assert sha(p)==h,p
    return spec


def build(cases,role):
    if role!='train':assert read(OUT/'models/frozen.json')['complete']
    expected=read(V25/'cache_audit_index.json');prior26=read(V26/'completion.json')['artifact_sha256'];hashes={}
    models={}
    for name,row in read(V25/'protocol.json')['models'].items():
        saved=torch.load(row['path'],map_location='cpu',weights_only=True)
        model=RetentionModel().eval().requires_grad_(False);model.load_state_dict(saved['state_dict']);models[name]=(model,saved)
    for ci,c in enumerate(cases,1):
        aa={}
        for arm in ['name','context_name']:
            f=prior.feature_path(c,arm);h=sha(f);hashes[str(f)]=h
            if role!='train':assert h==expected['development_feature_sha256'][str(f)]
            a,m=prior.load_npz(f);assert m['sample_id']==c['sample_id'] and m['arm']==arm
            aa[arm]=a
        a,b=aa['name'],aa['context_name'];points,ai,bi=union_points(a['points'],b['points']);n=len(points)
        scorefile=V26/'scores'/f'{prior.key(c["sample_id"])}.npz';assert sha(scorefile)==prior26[str(scorefile)];hashes[str(scorefile)]=sha(scorefile)
        scores,_=prior.load_npz(scorefile);values={}
        rb=dict(sample_id=c['sample_id'],n=len(b['points']),**b)
        for name,(model,saved) in models.items():
            sa=scores['A__'+name];assert len(sa)==len(a['points'])
            if rb['n']:
                inputs,_=tensors([rb],saved['mean'].numpy(),saved['std'].numpy(),'aligned','cpu')
                with torch.inference_mode():sb=model(**inputs).numpy()
            else:sb=np.empty(0,np.float32)
            values['A__'+name]=sa;values['B__'+name]=sb;values['U__'+name]=union_scores(sa,sb,ai,bi)
        features=np.zeros((n,3,48),np.float32);geometry=np.zeros((n,72),np.float32);counts=np.zeros((n,3),np.int64)
        provenance=[]
        for vi,v in enumerate(c['views']):
            af=Path(v['feature_path']);assert sha(af)==v['feature_sha256'];hashes[str(af)]=v['feature_sha256']
            with np.load(af,allow_pickle=False) as z:
                m=json.loads(str(z['metadata']));feat,_,valid=role_features(z['target_attention'],z['vision_hidden'],m['grid_shape'],points,c['image_size'],v['transform'])
                features[:,0]+=feat;counts[:,0]+=valid
            task=prior.task_for(c,vi,'context');bf=V25/f'raw/{task["task_id"]}.npz'
            h=sha(bf);assert h==expected['raw_sha256'][str(bf)];hashes[str(bf)]=h
            with np.load(bf,allow_pickle=False) as z:
                m=json.loads(str(z['metadata']));assert m['task']==task and m['transform']==v['transform']
                vision=z['vision_hidden'];attention=z['all_attention']
                for j,field in [(1,'target'),(2,'container')]:
                    ix=np.asarray(m['readout_stages'][field],int)-1
                    feat,geom,valid=role_features(attention[ix],vision,m['grid_shape'],points,c['image_size'],v['transform'])
                    features[:,j]+=feat;counts[:,j]+=valid
                    if j==2:geometry+=geom
                provenance.append(dict(view=vi,task_id=task['task_id'],role_indices={k:m['readout_stages'][k] for k in ['target','container']}))
        features/=np.maximum(counts[:,:,None],1);geometry/=np.maximum(counts[:,2,None],1)
        prior.npz(path(c),points=points,ai=ai,bi=bi,role=features,geometry=geometry,valid=counts>0,**values,
            metadata=np.asarray(json.dumps(dict(sample_id=c['sample_id'],cohort=c['cohort'],role_provenance=provenance,
                original_A_points_sha256=hashes[str(prior.feature_path(c,'name'))],original_B_points_sha256=hashes[str(prior.feature_path(c,'context_name'))],
                labels_used=False,weak_container_context_only=True))))
        if ci%4==0 or ci==len(cases):status('cached_fine_templates_'+role,completed=ci,total=len(cases))
    write(OUT/f'{role}_input_sha256.json',hashes)
    write(OUT/f'{role}_features_complete.json',dict(complete=True,at_utc=now(),cases=len(cases)))
    verify()


def load(c):return prior.load_npz(path(c))[0]


def record(c,a,model,arm,pool='protected',labels=None):
    scores=a['U__'+model];ix=retain(scores,a['ai'],protected=pool=='protected')
    role,geom,valid=a['role'],a['geometry'],a['valid']
    if arm=='shuffled':role,geom,valid,_=content_shuffle(role,geom,valid,a['ai'],c['sample_id'])
    x,base=probe_features(a['points'],c['image_size'],scores,a['ai'],a['bi'],role,geom,valid,ix,arm)
    r=dict(sample_id=c['sample_id'],source=c['source'],indices=ix,x=x,base=base)
    if labels is not None:r['y']=np.asarray([hit(p.tolist(),c,labels[c['sample_id']]) for p in a['points'][ix]],bool)
    return r


def accuracy(model,r):
    s=infer(model,r);j=int(s.argmax()) if len(s) else None
    return bool(j is not None and r['y'][j])


def identity_model():return dict(weight=np.zeros(DIM),mean=np.zeros(DIM),std=np.ones(DIM),iterations=0,rankable_queries=0,success=True)


def train(cases):
    assert all(c['cohort']=='train' for c in cases)
    data={c['sample_id']:load(c) for c in cases};labels=read(V25/'data/labels.json');frozen={}
    for model in NAMES:
        for arm in ARMS:
            rr=[record(c,data[c['sample_id']],model,arm,labels=labels) for c in cases];sources=sorted({r['source'] for r in rr})
            candidates=[]
            for ridge in [None,*RIDGE]:
                held=[];fits=[]
                for src in sources:
                    fit=[r for r in rr if r['source']!=src];val=[r for r in rr if r['source']==src]
                    m=identity_model() if ridge is None else fit_residual(fit,ridge)
                    fits.append(dict(source_excluded=src,success=m['success'],iterations=m['iterations'],rankable_queries=m['rankable_queries']))
                    held.extend(dict(sample_id=r['sample_id'],source=src,hit=accuracy(m,r)) for r in val)
                macro=float(np.mean([np.mean([r['hit'] for r in held if r['source']==src]) for src in sources]))
                candidates.append(dict(ridge=ridge,source_macro=macro,held_out=held,optimizer_folds=fits))
            best=max(candidates,key=lambda r:(round(r['source_macro'],12),float('inf') if r['ridge'] is None else r['ridge']))
            m=identity_model() if best['ridge'] is None else fit_residual(rr,best['ridge'])
            dest=OUT/f'models/{arm}__{model}.npz'
            prior.npz(dest,weight=m['weight'],mean=m['mean'],std=m['std'],metadata=np.asarray(json.dumps(dict(model=model,arm=arm,ridge=best['ridge']))))
            info=dict(model=model,arm=arm,ridge=best['ridge'],selected_OOF_macro=best['source_macro'],cv=candidates,
                optimizer={k:v for k,v in m.items() if k not in ('weight','mean','std')},path=str(dest),sha256=sha(dest),
                max_weight_abs=float(np.abs(m['weight']).max()),training_sample_ids=[c['sample_id'] for c in cases],development_used=False)
            frozen[arm+'__'+model]=info;write(dest.with_suffix('.json'),info)
            status('training_pairwise_residuals',model=model,arm=arm,ridge=best['ridge'],oof_macro=best['source_macro'],completed=len(frozen),total=30)
    write(OUT/'models/frozen.json',dict(complete=True,frozen_at_utc=now(),models=frozen,training_rgb=len(cases),development_used=False))
    status('all_residuals_frozen',models=len(frozen),training_rgb=len(cases))


def load_models():
    result={}
    for tag,r in read(OUT/'models/frozen.json')['models'].items():
        assert sha(r['path'])==r['sha256'];a,_=prior.load_npz(r['path']);result[tag]=a
    return result


def predict(cases):
    labels=read(V25/'data/labels.json');models=load_models();rows=[];coverage=[]
    old={(r['sample_id'],r['model'],r['arm']):r for r in read(V25/'predictions.json')['rows'] if r['regime']=='frozen'}
    parity=0
    for ci,c in enumerate(cases,1):
        a=load(c);sid=c['sample_id'];truth=np.asarray([hit(p.tolist(),c,labels[sid]) for p in a['points']],bool)
        for model in NAMES:
            for source,prefix,arm in [('name','A','A'),('context_name','B','B_frozen')]:
                raw=prior.load_record(c,source);scores=a[prefix+'__'+model];j=int(scores.argmax()) if len(scores) else None
                point=raw['points'][j].tolist() if j is not None else None
                assert point==old[sid,model,source]['point'];parity+=1
                rows.append(dict(sample_id=sid,model=model,arm=arm,point=point,hit=hit(point,c,labels[sid]),index=None))
            scores=a['U__'+model]
            pools={'A16':np.flatnonzero(a['ai']>=0)[np.argsort(-scores[a['ai']>=0],kind='stable')[:16]],
                'naive':retain(scores,a['ai'],protected=False),'protected':retain(scores,a['ai']),
                'A_complete':np.flatnonzero(a['ai']>=0),'B_complete':np.flatnonzero(a['bi']>=0),'union_complete':np.arange(len(scores))}
            for pool,ix in pools.items():coverage.append(dict(sample_id=sid,model=model,pool=pool,n=len(ix),oracle=bool(truth[ix].any()),
                supplemental_positive=bool((truth[ix]&(a['ai'][ix]<0)).any())))
            for pool,arm in [('naive','naive_frozen'),('protected','union_frozen')]:
                ix=pools[pool];j=int(ix[scores[ix].argmax()]) if len(ix) else None
                rows.append(dict(sample_id=sid,model=model,arm=arm,index=j,point=a['points'][j].tolist() if j is not None else None,
                                 hit=bool(j is not None and truth[j])))
            for arm in ARMS:
                for pool in (['protected','naive'] if arm=='full' else ['protected']):
                    r=record(c,a,model,arm,pool);s=infer(models[arm+'__'+model],r);local=int(s.argmax()) if len(s) else None
                    j=int(r['indices'][local]) if local is not None else None
                    rows.append(dict(sample_id=sid,model=model,arm=arm if pool=='protected' else 'full_naive',
                        index=j,point=a['points'][j].tolist() if j is not None else None,hit=bool(j is not None and truth[j])))
        if ci%6==0 or ci==len(cases):status('evaluating_frozen_residuals',completed=ci,total=len(cases),original_A_B_prediction_parity=parity)
    write(OUT/'predictions.json',dict(rows=rows,coverage=coverage,original_A_B_prediction_parity=parity))


def analyze(cases):
    pred=read(OUT/'predictions.json');lookup={(r['sample_id'],r['model'],r['arm']):r for r in pred['rows']}
    arms=['A','B_frozen','naive_frozen','union_frozen',*ARMS,'full_naive'];results={}
    for cohort in ['new_development','old_val','all']:
        cc=[c for c in cases if cohort=='all' or c['cohort']==cohort];ids={c['sample_id'] for c in cc};summary={};contrasts={};coverage={}
        for family in ['original3k','full']:
            summary[family]={};arrays={}
            for arm in arms:
                v=np.asarray([[lookup[c['sample_id'],f'{family}_s{s}',arm]['hit'] for c in cc] for s in [1,2,3]],float);arrays[arm]=v
                summary[family][arm]=dict(accuracy_percent=100*float(v.mean()),seed_hits=v.sum(1).astype(int).tolist(),
                    by_source={src:100*float(v[:,[i for i,c in enumerate(cc) if c['source']==src]].mean()) for src in sorted({c['source'] for c in cc})})
            for a,b in CONTRASTS:contrasts[family+'/'+a+'-'+b]=bootstrap(arrays[a]-arrays[b],cc,1-.05/16 if cohort=='new_development' else .95)
            coverage[family]={}
            for pool in ['A_complete','B_complete','union_complete','A16','naive','protected']:
                rr=[r for r in pred['coverage'] if r['sample_id'] in ids and r['model'].startswith(family+'_') and r['pool']==pool]
                coverage[family][pool]=dict(oracle_percent=100*float(np.mean([r['oracle'] for r in rr])),
                    mean_candidates=float(np.mean([r['n'] for r in rr])),seed_covered=[sum(r['oracle'] for r in rr if r['model']==f'{family}_s{s}') for s in [1,2,3]])
        results[cohort]=dict(n=len(cc),methods=summary,comparisons=contrasts,coverage=coverage)
    gates={}
    for family in ['original3k','full']:
        r=results['new_development']['comparisons'];old=results['old_val']['methods'][family]
        gates[family]=(r[family+'/full-A']['delta_pp']>=3 and all(r[family+'/'+k]['ci_pp'][0]>0 for k in ['full-A','full-prior','full-shuffled'])
            and old['full']['accuracy_percent']-old['A']['accuracy_percent']>=-5)
    write(OUT/'results.json',dict(complete=True,cohorts=results,positive_gates=gates,new_vlm_calls=0,independent_test=False,mainline_unchanged=True))
    lines=['V27：补充候选保留与注意力引导的细粒度视觉/候选比较。',
        '56原训练RGB、36历史复用开发RGB。原6评分器冻结，训练<=322系数线性残差。0新增VLM调用。',
        '等K16：朴素合并 vs A8+B新增8保护；固定保护池内消融评分。容器为弱上下文，不是任意锚点关系执行。',
        '模型/正则化只通过训练内来源留一选择，含零残差回退；开发前冻结。种子不是独立截图。']
    for co,r in results.items():
        lines+=['',f'{co},n={r["n"]}']
        for f,mm in r['methods'].items():
            for arm,m in mm.items():lines.append(f'{f}/{arm}: {m["accuracy_percent"]:.2f}%,命中{m["seed_hits"]}')
            lines.append('候选覆盖：'+json.dumps(r['coverage'][f],ensure_ascii=False))
        for k,v in r['comparisons'].items():lines.append(f'{k}: {v["delta_pp"]:+.2f}pp, CI{v["interval_percent"]:.4f}% {v["ci_pp"]}')
    lines+=['','进阶门槛：'+str(gates),'36张均为历史开发图，不构成独立泛化或MVP优势证明。']
    (OUT/'results.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8');status('analysis_complete',positive_gates=gates)


def audit(cases):
    verify();frozen=read(OUT/'models/frozen.json');freeze=datetime.fromisoformat(frozen['frozen_at_utc'])
    assert freeze<datetime.fromisoformat(read(OUT/'development_features_complete.json')['at_utc'])
    train={c['sample_id'] for c in cases if c['cohort']=='train'};dev=[c for c in cases if c['cohort']!='train']
    hashes={}
    for role in ['train','development']:
        for p,h in read(OUT/f'{role}_input_sha256.json').items():assert sha(p)==h,p;hashes[p]=h
    models=load_models();pred=read(OUT/'predictions.json');lookup={(r['sample_id'],r['model'],r['arm']):r for r in pred['rows']}
    labels=read(V25/'data/labels.json');replays=0;poolchecks=0
    for tag,r in frozen['models'].items():assert set(r['training_sample_ids'])==train and not r['development_used']
    for c in cases:
        a=load(c);oldA=prior.load_record(c,'name');oldB=prior.load_record(c,'context_name')
        p,ai,bi=union_points(oldA['points'],oldB['points']);np.testing.assert_array_equal(p,a['points'])
        assert np.array_equal(ai,a['ai']) and np.array_equal(bi,a['bi']);poolchecks+=1
        assert not any(k in c for k in ['target_geometry','bbox','label'])
    for c in dev:
        a=load(c);sid=c['sample_id'];truth=np.asarray([hit(p.tolist(),c,labels[sid]) for p in a['points']],bool)
        for model in NAMES:
            for arm in ARMS:
                for pool in (['protected','naive'] if arm=='full' else ['protected']):
                    r=record(c,a,model,arm,pool);s=infer(models[arm+'__'+model],r);i=int(s.argmax()) if len(s) else None
                    j=int(r['indices'][i]) if i is not None else None;row=lookup[sid,model,arm if pool=='protected' else 'full_naive']
                    assert j==row['index'] and bool(j is not None and truth[j])==row['hit'];replays+=1
    before=read(OUT/'registry_before.json');reg=read(ROOT/'CURRENT_RESEARCH.json')
    assert before['run_id']==reg['run_id'] and before['models']==reg['models']
    write(OUT/'final_audit.json',dict(complete=True,at_utc=now(),historical_cache_hashes_verified=len(hashes),
        candidate_union_replays=poolchecks,learned_prediction_replays=replays,original_A_B_prediction_parity=pred['original_A_B_prediction_parity'],
        model_freeze_before_development_features=True,training_only=True,new_vlm_calls=0,mainline_unchanged=True))
    artifacts={str(p):sha(p) for p in [OUT/n for n in ['protocol.json','results.json','results.txt','predictions.json','models/frozen.json','final_audit.json']]}
    for sub in ['features','models']:
        artifacts.update({str(p):sha(p) for p in (OUT/sub).glob('*.npz')})
    write(OUT/'completion.json',dict(complete=True,completed_at_utc=now(),artifact_sha256=artifacts,new_vlm_calls=0,mainline_unchanged=True))
    reg['active_experiment'].update(status='complete',phase='training_evaluation_analysis_and_audit_complete',results_path=str(OUT/'results.txt'),
        completion_path=str(OUT/'completion.json'),audit_path=str(OUT/'final_audit.json'),positive_gates=read(OUT/'results.json')['positive_gates'])
    reg['completed_experiments'].append(dict(run_id=OUT.name,status='complete',analysis_path=str(OUT/'results.txt'),mainline_unchanged=True))
    write(ROOT/'CURRENT_RESEARCH.json',reg)
    (OUT/'SESSION_STATE.txt').write_text('V27训练/评估/分析/核验完成。0新增VLM调用。历史主线不变。详细结论待人工整理。\n',encoding='utf-8')
    status('complete',learned_prediction_replays=replays,positive_gates=read(OUT/'results.json')['positive_gates'])


def main():
    torch.set_num_threads(4);start=time.perf_counter();prepare();cases=read(V25/'data/public.json')
    traincases=[c for c in cases if c['cohort']=='train'];dev=[c for c in cases if c['cohort']!='train']
    build(traincases,'train');train(traincases);build(dev,'development');predict(dev);analyze(dev);audit(cases)
    print(f'V27 main complete, seconds={time.perf_counter()-start:.1f}',flush=True)


if __name__=='__main__':
    try:main()
    except BaseException:
        OUT.mkdir(parents=True,exist_ok=True);(OUT/'failure.txt').write_text(traceback.format_exc(),encoding='utf-8');status('failed');raise
