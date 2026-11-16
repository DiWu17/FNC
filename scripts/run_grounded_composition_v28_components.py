"""Matched-pool scoring controls separating native geometry from visual states."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ.setdefault(k,'4')
from pathlib import Path
import sys,json,time,argparse
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import numpy as np
import run_grounded_composition_v28 as run
from clickagent.grounded_composition_v28 import fit,infer
from run_visual_feedback_v23 import read,write,sha,hit,bootstrap
OUT=run.OUT/'component_controls'
ARMS=['geometry_random','visual_random','geometry_combined','visual_combined']

def mask(record,arm):
    r=dict(record,x=record['x'].copy())
    if arm.startswith('geometry'):r['x'][:,:1152]=0;r['x'][:,1169:]=0
    else:r['x'][:,1152:]=0
    return r

def train():
    assert read(OUT/'plan.json')['source_sha256']==sha(Path(__file__))
    cases=[c for c in read(run.OUT/'data/public.json') if c['cohort']=='train'];labels=read(run.OUT/'data/labels.json')
    while not all(run.dest(c,'grounded').exists() for c in cases):
        if read(run.OUT/'status.json')['phase']=='failed':raise RuntimeError('Main failed')
        time.sleep(5)
    models={}
    for arm in ARMS:
        cc=[c for c in cases if arm.endswith('combined') or c['training_group']=='random']
        for name in run.NAMES:
            rr=[mask(run.record(c,name,'grounded',labels),arm) for c in cc];model=fit(rr,.3,True)
            path=OUT/f'models/{arm}__{name}.npz';run.npz(path,weight=model['weight'],mean=model['mean'],std=model['std'],metadata=np.asarray(json.dumps(dict(arm=arm,old_head=name))))
            models[arm+'__'+name]=dict(path=str(path),sha256=sha(path),new_training_rgb=len(cc),training_ids=[c['sample_id'] for c in cc],optimizer={k:v for k,v in model.items() if k not in ['weight','mean','std']})
            write(OUT/'progress.json',dict(completed=len(models),total=24));print(dict(arm=arm,old_head=name,completed=len(models),total=24),flush=True)
    write(OUT/'frozen.json',dict(complete=True,frozen_at_utc=run.now(),models=models,development_used=False))

def evaluate():
    assert read(OUT/'plan.json')['source_sha256']==sha(Path(__file__));frozen=read(OUT/'frozen.json');assert frozen['complete']
    cases=[c for c in read(run.OUT/'data/public.json') if c['cohort']!='train'];labels=read(run.OUT/'data/labels.json');models={}
    for tag,row in frozen['models'].items():assert sha(row['path'])==row['sha256'];models[tag]=run.load_npz(row['path'])[0]
    rows=[]
    for c in cases:
        for tag,m in models.items():
            arm,name=tag.split('__');r=mask(run.record(c,name,'grounded'),arm);j=int(infer(m,r).argmax()) if len(r['x']) else None;p=r['points'][j].tolist() if j is not None else None
            rows.append(dict(sample_id=c['sample_id'],arm=arm,old_head=name,point=p,hit=hit(p,c,labels[c['sample_id']])))
    lookup={(r['sample_id'],r['old_head'],r['arm']):r for r in rows};primary={(r['sample_id'],r['old_head'],r['arm']):r for r in read(run.OUT/'predictions_stage2.json')['rows']};results={}
    for cohort in ['new_development','old_val']:
        cc=[c for c in cases if c['cohort']==cohort];summaries={};contrasts={}
        for family in ['original3k','full']:
            arrays={arm:np.asarray([[lookup[c['sample_id'],f'{family}_s{s}',arm]['hit'] for c in cc] for s in [1,2,3]],float) for arm in ARMS}
            for arm in ['no_relation','grounded','combined']:arrays[arm]=np.asarray([[primary[c['sample_id'],f'{family}_s{s}',arm]['hit'] for c in cc] for s in [1,2,3]],float)
            summaries[family]={a:dict(percent=float(v.mean()*100),seed_hits=v.sum(1).astype(int).tolist()) for a,v in arrays.items()}
            for a,b in [('no_relation','geometry_random'),('grounded','geometry_random'),('combined','geometry_combined'),('combined','visual_combined')]:contrasts[family+'/'+a+'-'+b]=bootstrap(arrays[a]-arrays[b],cc,.99375 if cohort=='new_development' else .95)
        results[cohort]=dict(summary=summaries,contrasts=contrasts)
    write(OUT/'predictions.json',dict(rows=rows));write(OUT/'results.json',results)
    (OUT/'readout.txt').write_text('共享候选/观察/训练/损失/正则的评分组件对照；geometry仅原生坐标几何+旧先验，visual仅新目标视觉状态+旧先验。不能把复用候选中隐含的新坐标信息完全去掉。\n'+json.dumps(results,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    write(OUT/'completion.json',dict(complete=True,source_sha256=sha(Path(__file__)),artifact_sha256={str(p):sha(p) for p in OUT.rglob('*') if p.is_file() and p.name!='completion.json'}))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['register','wait_train','evaluate']);phase=parser.parse_args().phase
    if phase=='register':
        write(OUT/'plan.json',dict(registered_at_utc=run.now(),source_sha256=sha(Path(__file__)),description='Train-only fixed-ridge.3 same1184 coefficient architecture, exact same augmented pool/evidence. Zero visual+relation channels for geometry; zero geometry+relation for visual. random56 andcombined112. No tuning/selection/extraVLM. Preregistered before stage2fits or any new dev trajectory to distinguish benefits from actual visual states versus native-coordinate geometry.',arms=ARMS,statistics='8 secondary contrasts across2oldheadfamilies;99.375% source-stratified RGB bootstrap on reusednew72,95% old28. Exploratory component diagnosis, no independent superiority claim.',development_used=False,new_vlm_calls=0))
    elif phase=='wait_train':train()
    else:evaluate()
