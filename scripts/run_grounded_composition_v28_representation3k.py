"""Cached 2973-query representation scaling follow-up; no new VLM calls."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ.setdefault(k,'4')
from pathlib import Path
import sys,json,shutil
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import numpy as np
import run_grounded_composition_v28 as run
from clickagent.grounded_composition_v28 import fit,infer
from run_visual_feedback_v23 import read,write,sha,hit,bootstrap
MAIN=run.OUT;OUT=ROOT/'runs/grounded_composition_v28_representation3k';MODES=['raw','centered','shuffled']
run.OUT=OUT


def prepare():
    assert not (OUT/'protocol.json').exists()
    old=read(MAIN/'protocol.json');indices=read(run.V19/'data/indices.json')['train'];lookup={}
    manifests=[ROOT/'runs/multiregion_v13/data/train.jsonl',ROOT/'runs/quality_readout_v15_3k/data/new_train.jsonl']
    for path in manifests:
        for line in path.read_text(encoding='utf-8-sig').splitlines():
            c=json.loads(line);lookup[c['sample_id']]=c
    cc=[]
    for e in indices:
        c=lookup[e['sample_id']];p=read(e['pipeline_path']);r={k:c[k] for k in ['sample_id','dataset','image','image_size','image_sha256','instruction']}
        r.update(cohort='train',source=e['source'],group=e['group'],source_group=e['source_group'],training_group='all2973')
        r['views']=[dict(feature_path=p['global_feature_path'],feature_sha256=p['global_feature_sha256'],transform=None)]+[{k:v[k] for k in ['feature_path','feature_sha256','transform']} for v in p['local_views']]
        cc.append(r)
    dev=[c for c in read(MAIN/'data/public.json') if c['cohort']!='train'];assert len(cc)==2973 and not {c['group'] for c in cc}&{c['group'] for c in dev}
    write(OUT/'data/public.json',cc+dev);write(OUT/'data/indices.json',{e['sample_id']:e for e in indices})
    sealed=[Path(__file__),run.V19/'data/indices.json',MAIN/'protocol.json',MAIN/'predictions_stage1.json',*manifests,*run.SOURCES,OUT/'data/public.json',OUT/'data/indices.json']
    protocol=dict(registered_at_utc=run.now(),source_sha256=run.identity(),extension_source_sha256=sha(Path(__file__)),sealed_sha256={str(p):sha(p) for p in sealed},models=old['models'],
        motivation='Stage1 small56 negative is not enough to distinguish representation failure from inadequate residual supervision. Reuse ALL original2973 training queries/cached original-name views. Separate exploratory scaling follow-up, registered after56-stage results, before its own fitting or development scoring. No newVLM, independent test or adoption.',
        data=dict(training_queries=2973,unique_training_rgb=len({c['group'] for c in cc}),development_rgb=100,reused_development=True),
        design='Same A32protect6heads rule, same1152visual channels, same modes raw/centered/shuffled, sameold6priors, fixedridge.3, same query/source-balanced hard pair loss. Actual cached annotation vectors after feature construction; matched_count never treated as labels. All fits frozen before reused dev100scoring. Centered arm additionally3RGB-group folds diagnosis; no selection. All3 old fits/family reported.',
        costs='CPU cached extraction/fitting overlaps existingGPU observation job; zero newVLMcalls. Baselines/validation exactly originalV28A32features. No claim of pure3k totalbudget because shared pool uses all6oldheads.',
        comparisons=[['centered','raw'],['centered','shuffled'],['centered','small56_centered'],['centered','A']],statistics='8 exploratory contrasts across2headfamilies;99.375% source-stratifiedRGB bootstrap new72;95% old28. Not independent confirmation.')
    write(OUT/'protocol.json',protocol);shutil.copyfile(__file__,OUT/Path(__file__).name);run.status('registered_cached_representation3k',train_queries=2973,dev_rgb=100,new_vlm_calls=0)


def build(cases):
    for i,c in enumerate(cases,1):
        path=run.dest(c,'features')
        if path.exists():continue
        original,_=run.load(c,'original');points=original['points'];observations=[]
        for vi,v in enumerate(c['views']):
            assert sha(v['feature_path'])==v['feature_sha256']
            with np.load(v['feature_path'],allow_pickle=False) as z:observations.append((z['target_attention'],z['vision_hidden'],json.loads(str(z['metadata'])),v['transform'],vi>0))
        values={m:run.aggregate(c,points,m,observations)[0] for m in MODES}
        run.npz(path,points=points,**values,metadata=np.asarray(json.dumps(dict(sample_id=c['sample_id'],labels_used=False))))
        if i%16==0 or i==len(cases):run.status('cached_representation_features',completed=i,total=len(cases))


def record(c,mode,name,entries=None,cache=None):
    if c['cohort']=='train':
        a,original=cache[c['sample_id']] if cache is not None else (run.load(c,'features')[0],run.load(c,'original')[0])
    else:
        a=run.load_npz(MAIN/'representation'/f'{run.key(c["sample_id"])}.npz')[0];original=run.load_npz(MAIN/'original'/f'{run.key(c["sample_id"])}.npz')[0]
    s=original['scores'][run.NAMES.index(name)];base=(s-s.mean())/max(float(s.std()),1e-6) if len(s) else s
    r=dict(sample_id=c['sample_id'],source=c['source'],x=a[mode].astype(np.float64),base=base.astype(np.float64),points=a['points'],size=np.asarray(c['image_size']))
    if entries is not None:
        e=entries[c['sample_id']];assert sha(e['labels_path'])==e['labels_sha256'];old=np.load(e['labels_path']).astype(bool)
        _,meta=run.load(c,'original');r['y']=old[np.asarray(meta['indices'],int)]
    return r


def train(cases):
    entries=read(OUT/'data/indices.json');run.status('loading_3k_features_for_fit')
    cache={c['sample_id']:(run.load(c,'features')[0],run.load(c,'original')[0]) for c in cases};models={}
    folds={c['sample_id']:int(run.digest([c['group'],'v28_3k_fold'])[:16],16)%3 for c in cases}
    for mode in MODES:
        for name in run.NAMES:
            rr=[record(c,mode,name,entries,cache) for c in cases];oof=[]
            if mode=='centered':
                for fold in range(3):
                    m=fit([r for r in rr if folds[r['sample_id']]!=fold],.3,True)
                    oof.extend(dict(sample_id=r['sample_id'],hit=bool(r['y'][int(infer(m,r).argmax())]) if len(r['x']) else False) for r in rr if folds[r['sample_id']]==fold)
            m=fit(rr,.3,True);path=OUT/f'models/{mode}__{name}.npz'
            run.npz(path,weight=m['weight'],mean=m['mean'],std=m['std'],metadata=np.asarray(json.dumps(dict(mode=mode,old_head=name))))
            info=dict(path=str(path),sha256=sha(path),new_training_queries=len(cases),unique_training_rgb=len({c['group'] for c in cases}),train_hits=sum(bool(r['y'][int(infer(m,r).argmax())]) if len(r['x']) else False for r in rr),oof_hits=sum(r['hit'] for r in oof) if oof else None,optimizer={k:v for k,v in m.items() if k not in ['weight','mean','std']})
            models[mode+'__'+name]=info;run.status('cached_3k_fitting',mode=mode,old_head=name,completed=len(models),total=18,train_hits=info['train_hits'],oof_hits=info['oof_hits'])
    write(OUT/'models/frozen.json',dict(complete=True,frozen_at_utc=run.now(),models=models,development_used=False))


def evaluate(cases):
    labels=read(MAIN/'data/labels.json');frozen=read(OUT/'models/frozen.json');models={}
    for tag,row in frozen['models'].items():assert sha(row['path'])==row['sha256'];models[tag]=run.load_npz(row['path'])[0]
    rows=[]
    for c in cases:
        for tag,m in models.items():
            mode,name=tag.split('__');r=record(c,mode,name);j=int(infer(m,r).argmax()) if len(r['x']) else None;p=r['points'][j].tolist() if j is not None else None
            rows.append(dict(sample_id=c['sample_id'],mode=mode,old_head=name,point=p,hit=hit(p,c,labels[c['sample_id']])))
    lookup={(r['sample_id'],r['old_head'],r['mode']):r for r in rows};small={(r['sample_id'],r['old_head'],r['arm']):r for r in read(MAIN/'predictions_stage1.json')['rows']};results={}
    for co in ['new_development','old_val']:
        cc=[c for c in cases if c['cohort']==co];summary={};contrasts={}
        for family in ['original3k','full']:
            arrays={m:np.asarray([[lookup[c['sample_id'],f'{family}_s{s}',m]['hit'] for c in cc] for s in [1,2,3]],float) for m in MODES}
            for m,arm in [('A','A'),('small56_centered','repr_centered')]:arrays[m]=np.asarray([[small[c['sample_id'],f'{family}_s{s}',arm]['hit'] for c in cc] for s in [1,2,3]],float)
            summary[family]={a:dict(percent=float(v.mean()*100),seed_hits=v.sum(1).astype(int).tolist()) for a,v in arrays.items()}
            for a,b in read(OUT/'protocol.json')['comparisons']:contrasts[family+'/'+a+'-'+b]=bootstrap(arrays[a]-arrays[b],cc,.99375 if co=='new_development' else .95)
        results[co]=dict(summary=summary,contrasts=contrasts)
    write(OUT/'predictions.json',dict(rows=rows));write(OUT/'results.json',results)
    (OUT/'readout.txt').write_text('原2973训练查询的缓存视觉表示训练；对比此前56例；同1152系数/候选/损失/正则。探索性复用开发结果，不是独立确认。\n'+json.dumps(results,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def main():
    if not (OUT/'protocol.json').exists():prepare()
    p=run.verify();assert p['extension_source_sha256']==sha(Path(__file__))
    cases=read(OUT/'data/public.json');train_cases=[c for c in cases if c['cohort']=='train'];dev=[c for c in cases if c['cohort']!='train']
    run.score_original(train_cases);build(train_cases);train(train_cases);evaluate(dev);run.verify()
    files=[p for p in OUT.rglob('*') if p.is_file() and p.suffix in ['.json','.npz','.txt','.py'] and p.name not in ['status.json','completion.json']]
    write(OUT/'completion.json',dict(complete=True,artifact_sha256={str(p):sha(p) for p in files},new_vlm_calls=0));run.status('complete',new_training_queries=2973,new_vlm_calls=0)

if __name__=='__main__':
    try:main()
    except BaseException as e:
        import traceback
        run.status('failed',error=repr(e),traceback=traceback.format_exc());raise
