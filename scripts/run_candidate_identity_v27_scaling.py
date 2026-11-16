"""Recorded normalization correction; original V27 sources/results stay sealed."""
import os
for k,v in {'PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4','MKL_NUM_THREADS':'4'}.items():os.environ.setdefault(k,v)
from pathlib import Path
from datetime import datetime,timezone
import json,sys,shutil,time,traceback
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import numpy as np
import torch
import run_candidate_identity_v27 as base
from clickagent.candidate_identity_v27 import ARMS,RIDGE,DIM,fit_residual,infer,retain
from run_visual_feedback_v23 import read,write,sha,hit,bootstrap
OUT=base.OUT/'scaling_control'
ROUNDTRIPS=[]
PAIRS=[('full','A'),('visual','A'),('visual','prior'),('full','prior'),
       ('full','pair_visual'),('full','shuffled'),('full','old_full'),('visual','old_visual')]


def now():return datetime.now(timezone.utc).isoformat()
def status(phase,**kw):
    r=dict(phase=phase,complete=phase=='complete',updated_at_utc=now(),**kw)
    write(OUT/'status.json',r);print(json.dumps(r,ensure_ascii=False),flush=True)


def scaled_fit(records,ridge):
    x=np.concatenate([r['x'] for r in records if len(r['x'])]);sigma=x.std(0)
    scale=np.where(sigma>1e-10,np.maximum(sigma,1e-4),1.)
    transformed=[dict(r,x=r['x']/scale) for r in records]
    inner=fit_residual(transformed,ridge)
    m=dict(inner,mean=inner['mean']*scale,std=inner['std']*scale)
    error=max(float(np.max(np.abs(infer(inner,s)-infer(m,r)))) if len(r['x']) else 0
              for r,s in zip(records,transformed))
    assert error<1e-8,error;ROUNDTRIPS.append(error)
    m['normalization_roundtrip_error']=error
    return m


def prepare():
    assert not (OUT/'protocol.json').exists()
    # Synthetic verification of train-scaling/inverse-scaling equivalence.
    x=np.zeros((3,DIM));x[:,0]=[1e-3,0,-1e-3];x[:,1]=[1,0,-1]
    r=dict(x=x,base=np.zeros(3),y=np.asarray([True,False,False]),source='synthetic')
    m=scaled_fit([r],.3);assert infer(m,r).argmax()==0
    completion=read(base.OUT/'completion.json')
    for p,h in completion['artifact_sha256'].items():assert sha(p)==h,p
    paths=[Path(__file__),ROOT/'src/clickagent/candidate_identity_v27.py',ROOT/'scripts/run_candidate_identity_v27.py',
           base.OUT/'protocol.json',base.OUT/'completion.json',base.OUT/'models/frozen.json',base.OUT/'predictions.json',
           base.V25/'data/public.json',base.V25/'data/labels.json']
    spec=dict(recorded_at_utc=now(),source_sha256=sha(Path(__file__)),sealed_sha256={str(p):sha(p) for p in paths},
        reason='Posthoc implementation-setting correction after V27: floor std=.05 suppressed visual/peer channels(std median .00149/.00083) relative to geometry(.219). Original results remain visible; cannot attribute original negative solely to unusable visual evidence.',
        change='Use training-only per-channel std, floor1e-4, constant channels scale1. Reparameterize original fitter and verify raw-v-transformed logits within1e-8. Original candidate sets,322 features,base logits,loss,ridge grid,LOSO selection,identity fallback unchanged.',
        data='Same56 original-training and36 reused development. All new coefficients frozen before corrected development evaluation. No new VLM/image encoding.',
        transparency='This correction is motivated after seeing the primary negative outcome. It is exploratory follow-up, not fresh independent confirmation; all5 arms and all3 fits/family reported. No development-based selection of normalization or checkpoint.',
        comparisons=PAIRS,statistics='5000 source-stratified RGB bootstrap;16 contrasts across2 families,99.6875% intervals on new24,95% old12/all36.',
        original_results_unchanged=True,mainline_unchanged=True,new_vlm_calls=0)
    write(OUT/'protocol.json',spec);write(OUT/'synthetic_verification.json',dict(passed=True,roundtrip_error=ROUNDTRIPS[-1]))
    dest=OUT/'source_archive'/Path(__file__).name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(__file__,dest)
    reg=read(ROOT/'CURRENT_RESEARCH.json');reg['active_experiment'].update(status='running',phase='normalization_control_running',status_path=str(OUT/'status.json'),normalization_protocol_path=str(OUT/'protocol.json'))
    write(ROOT/'CURRENT_RESEARCH.json',reg);status('registered_normalization_correction')


def verify():
    for p,h in read(OUT/'protocol.json')['sealed_sha256'].items():assert sha(p)==h,p


def train(cases):
    labels=read(base.V25/'data/labels.json');data={c['sample_id']:base.load(c) for c in cases};models={}
    for name in base.NAMES:
        for arm in ARMS:
            records=[base.record(c,data[c['sample_id']],name,arm,labels=labels) for c in cases]
            sources=sorted({r['source'] for r in records});choices=[]
            for ridge in [None,*RIDGE]:
                held=[]
                for src in sources:
                    fit=[r for r in records if r['source']!=src];val=[r for r in records if r['source']==src]
                    m=base.identity_model() if ridge is None else scaled_fit(fit,ridge)
                    held.extend(dict(sample_id=r['sample_id'],source=src,hit=base.accuracy(m,r)) for r in val)
                macro=float(np.mean([np.mean([r['hit'] for r in held if r['source']==src]) for src in sources]))
                choices.append(dict(ridge=ridge,source_macro=macro,held_out=held))
            best=max(choices,key=lambda r:(round(r['source_macro'],12),float('inf') if r['ridge'] is None else r['ridge']))
            m=base.identity_model() if best['ridge'] is None else scaled_fit(records,best['ridge'])
            dest=OUT/f'models/{arm}__{name}.npz'
            base.prior.npz(dest,weight=m['weight'],mean=m['mean'],std=m['std'],metadata=np.asarray(json.dumps(dict(model=name,arm=arm))))
            row=dict(path=str(dest),sha256=sha(dest),model=name,arm=arm,ridge=best['ridge'],selected_OOF=best['source_macro'],
                choices=choices,training_ids=[c['sample_id'] for c in cases],optimizer={k:v for k,v in m.items() if k not in ('weight','mean','std')})
            models[arm+'__'+name]=row
            status('corrected_training_only_selection',model=name,arm=arm,ridge=best['ridge'],oof=best['source_macro'],completed=len(models),total=30)
    write(OUT/'models/frozen.json',dict(complete=True,frozen_at_utc=now(),models=models,development_used=False))


def load_models():
    result={}
    for tag,r in read(OUT/'models/frozen.json')['models'].items():
        assert sha(r['path'])==r['sha256'];result[tag]=base.prior.load_npz(r['path'])[0]
    return result


def evaluate(cases):
    models=load_models();labels=read(base.V25/'data/labels.json');rows=[]
    for c in cases:
        a=base.load(c)
        for name in base.NAMES:
            for arm in ARMS:
                r=base.record(c,a,name,arm);s=infer(models[arm+'__'+name],r);j=int(r['indices'][s.argmax()]) if len(s) else None
                point=a['points'][j].tolist() if j is not None else None
                rows.append(dict(sample_id=c['sample_id'],model=name,arm=arm,index=j,point=point,hit=hit(point,c,labels[c['sample_id']])))
    write(OUT/'predictions.json',dict(rows=rows,evaluated_at_utc=now()));status('corrected_evaluation_complete',rows=len(rows))


def analyze(cases):
    old={(r['sample_id'],r['model'],r['arm']):r for r in read(base.OUT/'predictions.json')['rows']}
    rows=read(OUT/'predictions.json')['rows'];lookup={(r['sample_id'],r['model'],r['arm']):r for r in rows};result={}
    for cohort in ['new_development','old_val','all']:
        cc=[c for c in cases if cohort=='all' or c['cohort']==cohort];summary={};contrasts={}
        for family in ['original3k','full']:
            summary[family]={};arrays={}
            for arm in [*ARMS,'A','old_full','old_visual']:
                origin=old if arm=='A' or arm.startswith('old_') else lookup;key=arm[4:] if arm.startswith('old_') else arm
                v=np.asarray([[origin[c['sample_id'],f'{family}_s{s}',key]['hit'] for c in cc] for s in [1,2,3]],float)
                arrays[arm]=v;summary[family][arm]=dict(accuracy_percent=100*float(v.mean()),seed_hits=v.sum(1).astype(int).tolist(),
                    by_source={src:100*float(v[:,[i for i,c in enumerate(cc) if c['source']==src]].mean()) for src in sorted({c['source'] for c in cc})})
            for a,b in PAIRS:contrasts[family+'/'+a+'-'+b]=bootstrap(arrays[a]-arrays[b],cc,1-.05/16 if cohort=='new_development' else .95)
        result[cohort]=dict(n=len(cc),methods=summary,comparisons=contrasts)
    write(OUT/'results.json',dict(complete=True,cohorts=result,exploratory_posthoc=True,independent_test=False,new_vlm_calls=0))
    lines=['V27归一化设置修正：按训练内实际尺度平衡视觉与几何特征；原结果不覆盖。']
    for co,r in result.items():
        lines+=['',co]
        for f,mm in r['methods'].items():
            for arm,m in mm.items():lines.append(f'{f}/{arm}: {m["accuracy_percent"]:.2f}%,{m["seed_hits"]}')
        for k,v in r['comparisons'].items():lines.append(f'{k}: {v["delta_pp"]:+.2f}pp CI{v["interval_percent"]:.4f}%{v["ci_pp"]}')
    (OUT/'results.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def audit(cases):
    verify();f=read(OUT/'models/frozen.json');p=read(OUT/'predictions.json')
    assert datetime.fromisoformat(f['frozen_at_utc'])<datetime.fromisoformat(p['evaluated_at_utc'])
    models=load_models();lookup={(r['sample_id'],r['model'],r['arm']):r for r in p['rows']};labels=read(base.V25/'data/labels.json');checks=0
    for c in cases:
        a=base.load(c)
        for name in base.NAMES:
            for arm in ARMS:
                r=base.record(c,a,name,arm);s=infer(models[arm+'__'+name],r);j=int(r['indices'][s.argmax()]) if len(s) else None
                row=lookup[c['sample_id'],name,arm];point=a['points'][j].tolist() if j is not None else None
                assert row['index']==j and row['point']==point and row['hit']==hit(point,c,labels[c['sample_id']]);checks+=1
    write(OUT/'final_audit.json',dict(complete=True,at_utc=now(),decision_replays=checks,normalization_roundtrip_checks=len(ROUNDTRIPS),
        max_roundtrip_error=max(ROUNDTRIPS),training_only_selection=True,models_frozen_before_evaluation=True,original_experiment_unchanged=True))
    paths=[OUT/n for n in ['protocol.json','predictions.json','results.json','results.txt','models/frozen.json','final_audit.json','synthetic_verification.json']]
    files={str(p):sha(p) for p in paths};files.update({str(p):sha(p) for p in (OUT/'models').glob('*.npz')})
    write(OUT/'completion.json',dict(complete=True,at_utc=now(),artifact_sha256=files,new_vlm_calls=0))
    reg=read(ROOT/'CURRENT_RESEARCH.json');reg['active_experiment'].update(status='complete',phase='primary_and_normalization_control_complete',
        normalization_results_path=str(OUT/'results.txt'),normalization_completion_path=str(OUT/'completion.json'))
    write(ROOT/'CURRENT_RESEARCH.json',reg);status('complete',decision_replays=checks,normalization_roundtrip_checks=len(ROUNDTRIPS))


def main():
    torch.set_num_threads(4);start=time.perf_counter();prepare();cases=read(base.V25/'data/public.json')
    train([c for c in cases if c['cohort']=='train']);dev=[c for c in cases if c['cohort']!='train']
    evaluate(dev);analyze(dev);audit(dev);print(f'Normalization control complete, seconds={time.perf_counter()-start:.1f}',flush=True)


if __name__=='__main__':
    try:main()
    except BaseException:
        OUT.mkdir(parents=True,exist_ok=True);(OUT/'failure.txt').write_text(traceback.format_exc(),encoding='utf-8');status('failed');raise
