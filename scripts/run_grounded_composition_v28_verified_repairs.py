"""Separate, train-only verified-correction follow-up after mining-field erratum.

Uses the already planned observations; preserves every original V28 arm.
"""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ.setdefault(k,'4')
import sys,json,time,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
import run_grounded_composition_v28 as run
from run_visual_feedback_v23 import read,write,sha,hit,bootstrap
from clickagent.grounded_composition_v28 import infer,identity
OUT=run.OUT/'verified_repairs'


def fit_priority(rr):
    if not rr:return identity(1184)
    xx=np.concatenate([r['x'] for r in rr if len(r['x'])]);sigma=xx.std(0);mean=xx.mean(0);std=np.where(sigma>1e-10,np.maximum(sigma,1e-4),1.)
    sources=sorted({r['source'] for r in rr});counts={s:sum(r['source']==s for r in rr) for s in sources};ds=[];bs=[];ws=[]
    for r in rr:
        pos=np.flatnonzero(r['y']);neg=np.flatnonzero(~r['y'])
        if not len(pos) or not len(neg):continue
        native=pos[pos>=r['old_n']];other=pos[pos<r['old_n']]
        pos=np.r_[native[np.argsort(-r['base'][native],kind='stable')],other[np.argsort(-r['base'][other],kind='stable')]][:4]
        distance=np.linalg.norm(r['points'][neg,None]/r['size']-r['points'][pos][None]/r['size'],axis=-1).min(1)
        neg=np.unique(np.r_[neg[np.argsort(distance,kind='stable')[:8]],neg[np.argsort(-r['base'][neg],kind='stable')[:8]]])
        pi=np.repeat(pos,len(neg));ni=np.tile(neg,len(pos));ds.append((r['x'][pi]-r['x'][ni])/std);bs.append(r['base'][pi]-r['base'][ni]);ws.append(np.full(len(pi),1/(len(sources)*counts[r['source']]*len(pi))))
    if not ds:return identity(1184)
    d=np.concatenate(ds);b=np.concatenate(bs);w=np.concatenate(ws);w/=w.sum();ridge=.3
    def obj(theta):
        margin=b+d@theta
        return float(w@np.logaddexp(0,-margin)+ridge*(theta@theta)/2),-d.T@(w*expit(-margin))+ridge*theta
    result=minimize(obj,np.zeros(1184),jac=True,method='L-BFGS-B',options=dict(maxiter=160,ftol=1e-9,gtol=1e-5))
    assert np.isfinite(result.x).all() and (result.success or np.max(np.abs(result.jac))<1e-4)
    return dict(weight=result.x,mean=mean,std=std,success=bool(result.success),rankable_queries=len(ds),pairs=len(d),iterations=int(result.nit))


def train():
    assert read(OUT/'plan.json')['source_sha256']==sha(Path(__file__))
    cc=[c for c in read(run.OUT/'data/public.json') if c['cohort']=='train'];labels=read(run.OUT/'data/labels.json');mining={r['sample_id']:r for r in read(run.OUT/'data/training_mining_corrected.json')}
    while not all(run.dest(c,'grounded').exists() for c in cc):
        if read(run.OUT/'status.json')['phase']=='failed':raise RuntimeError('Main V28 failed')
        time.sleep(5)
    eligible=[];details=[]
    for c in cc:
        a,_=run.load(c,'original');g,_=run.load(c,'grounded');old_n=len(a['points'])
        truth=np.asarray([hit(p.tolist(),c,labels[c['sample_id']]) for p in g['points']],bool)
        good=bool(truth[old_n:].any()) and mining[c['sample_id']]['original_failed_seeds']>=2
        if good:eligible.append(c)
        details.append(dict(sample_id=c['sample_id'],source=c['source'],eligible=good,native_correct=bool(truth[old_n:].any()),old_failed_seeds=mining[c['sample_id']]['original_failed_seeds']))
    chosen=[];matched=[]
    for source in sorted({c['source'] for c in cc}):
        possible=sorted([c for c in eligible if c['source']==source],key=lambda c:run.digest([c['sample_id'],'v28_verified']))[:6]
        chosen+=possible
        matched+=sorted([c for c in cc if c['source']==source and c['training_group']=='random'],key=lambda c:run.digest([c['sample_id'],'v28_verified_random']))[:len(possible)]
    assert len(chosen)==len(matched)
    datasets=dict(verified_repair=chosen,matched_random=matched,all112_priority=cc);write(OUT/'mining.json',dict(eligible=details,datasets={a:[c['sample_id'] for c in v] for a,v in datasets.items()},development_used=False))
    models={}
    for arm,subset in datasets.items():
        fold={}
        for src in sorted({c['source'] for c in subset}):
            for j,c in enumerate(sorted([c for c in subset if c['source']==src],key=lambda c:run.digest([c['group'],'verified_fold']))):fold[c['sample_id']]=j%3
        for name in run.NAMES:
            records=[]
            for c in subset:
                r=run.record(c,name,'grounded',labels);r['old_n']=len(run.load(c,'original')[0]['points']);records.append(r)
            oof=[]
            for f in range(3):
                fit_rr=[r for r in records if fold[r['sample_id']]!=f];val=[r for r in records if fold[r['sample_id']]==f];m=fit_priority(fit_rr)
                oof.extend(dict(sample_id=r['sample_id'],hit=bool(r['y'][int(infer(m,r).argmax())]) if len(r['x']) else False) for r in val)
            m=fit_priority(records);path=OUT/f'models/{arm}__{name}.npz';run.npz(path,weight=m['weight'],mean=m['mean'],std=m['std'],metadata=np.asarray(json.dumps(dict(arm=arm,old_head=name))))
            models[arm+'__'+name]=dict(path=str(path),sha256=sha(path),new_training_rgb=len(subset),training_ids=[c['sample_id'] for c in subset],oof=oof,optimizer={k:v for k,v in m.items() if k not in ['weight','mean','std']})
            print(dict(arm=arm,old_head=name,n=len(subset),completed=len(models),total=18),flush=True);write(OUT/'progress.json',dict(phase='training',completed=len(models),total=18))
    write(OUT/'frozen.json',dict(complete=True,frozen_at_utc=run.now(),models=models,development_used=False))


def evaluate():
    assert read(OUT/'plan.json')['source_sha256']==sha(Path(__file__));frozen=read(OUT/'frozen.json');assert frozen['complete']
    dev=[c for c in read(run.OUT/'data/public.json') if c['cohort']!='train'];labels=read(run.OUT/'data/labels.json');models={}
    for tag,r in frozen['models'].items():assert sha(r['path'])==r['sha256'];models[tag]=run.load_npz(r['path'])[0]
    rows=[]
    for c in dev:
        for tag,m in models.items():
            arm,name=tag.split('__');r=run.record(c,name,'grounded');j=int(infer(m,r).argmax()) if len(r['x']) else None;p=r['points'][j].tolist() if j is not None else None
            rows.append(dict(sample_id=c['sample_id'],arm=arm,old_head=name,point=p,hit=hit(p,c,labels[c['sample_id']])))
    lookup={(r['sample_id'],r['old_head'],r['arm']):r for r in rows};primary={(r['sample_id'],r['old_head'],r['arm']):r for r in read(run.OUT/'predictions_stage2.json')['rows']};results={}
    for co in ['new_development','old_val']:
        cc=[c for c in dev if c['cohort']==co];summary={};contrasts={}
        for family in ['original3k','full']:
            arrays={arm:np.asarray([[lookup[c['sample_id'],f'{family}_s{s}',arm]['hit'] for c in cc] for s in [1,2,3]],float) for arm in ['verified_repair','matched_random','all112_priority']}
            arrays['primary_combined']=np.asarray([[primary[c['sample_id'],f'{family}_s{s}','combined']['hit'] for c in cc] for s in [1,2,3]],float)
            summary[family]={a:dict(percent=float(v.mean()*100),seed_hits=v.sum(1).astype(int).tolist()) for a,v in arrays.items()}
            for a,b in [('verified_repair','matched_random'),('all112_priority','primary_combined')]:contrasts[family+'/'+a+'-'+b]=bootstrap(arrays[a]-arrays[b],cc,.9875 if co=='new_development' else .95)
        results[co]=dict(summary=summary,contrasts=contrasts)
    first_dev=min(run.dest(c,'raw/global').stat().st_mtime_ns for c in dev);from datetime import datetime
    frozen_ns=int(datetime.fromisoformat(frozen['frozen_at_utc']).timestamp()*1e9)
    results['frozen_before_first_development_output']=frozen_ns<first_dev
    write(OUT/'predictions.json',dict(rows=rows));write(OUT/'results.json',results)
    (OUT/'readout.txt').write_text('更正后独立登记的训练标签核验补充正例实验；同数量/来源分布随机对照；不替换原有任何分支。\n'+json.dumps(results,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    write(OUT/'completion.json',dict(complete=True,source_sha256=sha(Path(__file__)),artifact_sha256={str(p):sha(p) for p in OUT.rglob('*') if p.is_file() and p.name!='completion.json'}))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['register','wait_train','evaluate']);phase=parser.parse_args().phase
    if phase=='register':
        write(OUT/'plan.json',dict(registered_at_utc=run.now(),source_sha256=sha(Path(__file__)),reason='Separate correction-supervision experiment after matched_count semantic erratum; main112 sample IDs and all primary arms/results preserved.',eligibility='TRAIN ONLY original3k old selected wrong >=2/3 AND correct newly added native point exists in actual label-free augmented pool. At most6/source by fixed SHA. Matched random draws SAME count/source from original random56. Count may be0; report insufficiency, never choose from development.',arms=['verified_repair','matched_random','all112_priority'],learning='Same1184-dimensional grounded features, same per-train-point normalization, ridge.3, query/source-balanced pair loss/hard negatives. Prefer up to3 actually correct new native points in top4 training positives, fill with old positives. No annotated points added to inference. Fixed3 RGB folds diagnostic only. Compare verified v equal-size random; all112 native-priority v originalcombined112. No selection.',statistics='4 secondary contrasts across2 old-head families;98.75% source-stratified RGB bootstrap on reused new72. Separate from original20 primary contrasts; exploratory train-only follow-up, no independent claim.',new_vlm_calls=0,registered_before_stage2_fitting=True,development_used=False))
    elif phase=='wait_train':train()
    else:evaluate()
