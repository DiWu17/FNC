"""Training-only positive-candidate supplement;fixed natural DEV candidates."""
from collections import Counter,defaultdict
from pathlib import Path
import argparse
import shutil
import sys
import numpy as np
from PIL import Image
import run_reinspection_v31 as run
import run_context256_v31 as context
from clickagent.reinspection_v31 import (MODES,unit,joint_vector,pair_geometry,fit as original_fit,
    stack,kernel,scores,decide)

PARENT=run.OUT;CONTEXT=PARENT/'context256';OUT=PARENT/'positive_utility'

def fit_rgb(records,mode,ridge=.01):
    x=stack(records,mode,'train');y=np.asarray([r['delta'] for r in records],float)
    source_rgb=defaultdict(set);pair_counts=Counter()
    for r in records:source_rgb[r['source']].add(r['rgb_id']);pair_counts[r['rgb_id']]+=1
    w=np.asarray([1/(len(source_rgb)*len(source_rgb[r['source']])*pair_counts[r['rgb_id']]) for r in records])
    assert np.isclose(w.sum(),1.)
    alpha=np.linalg.solve(kernel(x,x,mode)+np.diag(ridge/w),y)
    return dict(mode=mode,ridge=ridge,basis=x,alpha=alpha,training_ids=[r['sample_id'] for r in records],
        training_rgb=len(pair_counts),training_pairs=len(records),repairs=int((y>0).sum()),harms=int((y<0).sum()),ties=int((y==0).sum()),
        feature_dimension=4096,geometry_dimension=7,weighting='Each RGB has same total source-balanced weight regardless of number of pairs')

def preflight():
    rr=[dict(sample_id=str(i),rgb_id=str(i),source='s',eligible=True,delta=delta,
        content=np.asarray([float(delta),.2]),blurred=np.asarray([.1,.2]),geometry=np.asarray([1.,float(delta)])) for i,delta in enumerate([-1,0,1])]
    a=fit_rgb(rr,'combined');b=original_fit(rr,'combined')
    np.testing.assert_allclose(scores(a,rr),scores(b,rr),atol=1e-10)
    duplicate=rr+[dict(rr[0],sample_id='0duplicate')];d=fit_rgb(duplicate,'combined')
    np.testing.assert_allclose(scores(a,rr),scores(d,rr),atol=1e-10)
    return dict(complete=True,single_pair_matches_original=True,duplicating_same_rgb_pair_does_not_increase_weight=True)

def prepare():
    if (OUT/'protocol.json').exists():return
    assert not (PARENT/'predictions.json').exists(),'Register before any parent development scoring'
    parent=run.verify();mined=run.read(PARENT/'training_positive_mining.json')
    assert mined['training_only'] and not mined['development_used'] and len(mined['rows'])==20
    # Correct the descriptive count:existing training RGB gained a useful positive,
    # no independent image was added by creating a second pair on it.
    mined.pop('added_independent_training_queries',None)
    mined.update(additional_positive_query_groups=20,new_training_rgb=0,original_training_rgb=112)
    run.write(PARENT/'training_positive_mining.json',mined)
    OUT.mkdir(parents=True,exist_ok=True);run.write(OUT/'preflight.json',preflight())
    for name in ['public','labels']:
        p=OUT/'data'/f'{name}.json';p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(PARENT/'data'/f'{name}.json',p)
    for name in ['baseline_input_index.json','mvp_input_index.json']:shutil.copyfile(PARENT/name,OUT/name)
    shutil.copyfile(PARENT/'training_positive_mining.json',OUT/'mining.json')
    deps=[Path(__file__),run.ROOT/'scripts/run_context256_v31.py',PARENT/'protocol.json',CONTEXT/'protocol.json',OUT/'mining.json',OUT/'preflight.json',*map(Path,parent['hashes'])]
    spec=dict(registered_at_utc=run.now(),parent_stage_at_registration=run.read(PARENT/'status.json'),
        trigger='TRAIN only:97natural eligible queries,6repair/26harm/65tie. Mine actual correct predicted alternatives on20existingTRAIN queries whose original pair cannot repair.',
        supervision='No new RGB. Add at most ONEpositive pair per selected trainingRGB. Base stays same natural prediction;alternative highest-scored GT-correct existing new-global proposer candidate,else highest old6-head-mean-z GT-correct A32 candidate. No annotated center/box injected as point. No DEVlabel/point used for mining. Candidate selection by TRAIN labels is supervision,not an inference algorithm.',
        inputs='Same256px real-context canvas,fullinstruction,4096Djointprefill,clear/blur,bothorders,geometry/content/combined/blurred/shuffled heads. Original natural cases/features copied fromcontext256. All additional prefills only TRAIN. DEV remains exact same natural pairs and observations.',
        fitting='Weighted kernelridge.01,thresholdstrict>.1 unchanged. Each source/RGB has equal total weight;multiple pairs on sameRGB divide thatweight. 3fold source_groupCV,never put natural and mined pair ofoneRGB in different folds. Main CV prediction targets natural pairs;auxiliary mined-positive validation separate. No hyperparameter/thresholdsearch.',
        statistics='3supplemental all100comparisons98.333%source/RGBbootstrap:aug_combined-context256_combined,aug_combined-aug_blurred,aug_combined-short_single. New72/old28secondary. All arms/negative results reported. Synthetic positive sampling changes rival distribution;natural DEV decides net benefit. No promotion.',
        frozen_before='Finalnewheads frozen before DEV replay;reuse cached DEVobservations generated by primary/context experiments,not newpost-freezeDEVinference.',
        hashes={str(p):run.sha(p) for p in deps},backbone_inventory=parent['backbone_inventory'])
    run.write(OUT/'protocol.json',spec);print('Registered20-query training-positive supplement before parentDEVscoring.',flush=True)

def verify():
    spec=run.read(OUT/'protocol.json')
    for p,h in spec['hashes'].items():assert run.sha(p)==h,p
    return spec

def extra_path(c):return OUT/'augmented_features'/(run.key(c['sample_id'])+'.npz')

def collect_extra(rt,c,mined):
    a,m=run.load(CONTEXT/'features'/(run.key(c['sample_id'])+'.npz'))
    _,gm=run.load(PARENT/'raw/global'/(run.key(c['sample_id'])+'.npz'));_,fm=run.load(PARENT/'raw/local0'/(run.key(c['sample_id'])+'.npz'))
    base,alternative=m['base'],mined['positive_point'];assert base!=alternative
    with Image.open(c['image']) as source:image=source.convert('RGB')
    observations={}
    try:
        for blur in [False,True]:
            for reverse in [False,True]:
                name=f'{"blur" if blur else "clear"}_{int(reverse)}';p=OUT/'augmented_judge'/name/(run.key(c['sample_id'])+'.npz')
                if p.exists():observations[name]=run.load(p);continue
                left,right=(alternative,base) if reverse else (base,alternative)
                canvas,g=context.canvas256(image,left,right,blur)
                try:hidden,meta=rt.judge(canvas,c['instruction'])
                finally:canvas.close()
                meta.update(sample_id=c['sample_id'],base=base,alternative=alternative,geometry=g,reverse=reverse,blur=blur,training_only=True)
                run.npz(p,dict(hidden=hidden),meta);observations[name]=(dict(hidden=hidden),meta)
    finally:image.close()
    content=joint_vector(observations['clear_0'][0]['hidden'],observations['clear_1'][0]['hidden'])
    blurred=joint_vector(observations['blur_0'][0]['hidden'],observations['blur_1'][0]['hidden'])
    geo=pair_geometry(base,alternative,gm['global_point'],m['first_crop'],m['diverse_crop'],c['image_size'],
        dict(base=fm['target'],alternative=gm['target'] if mined['origin']=='new_global_proposer' else None,global_=gm['target'])|{'global':gm['target']})
    meta=dict(m,sample_id=c['sample_id']+'|mined_positive',rgb_id=c['sample_id'],base=base,alternative=alternative,
              eligible=True,training_only=True,mined_origin=mined['origin'])
    run.npz(extra_path(c),dict(content=content,blurred=blurred,geometry=geo),meta)

def train(cases):
    labels=run.read(OUT/'data/labels.json');rr=run.records(cases,labels)
    natural=[dict(r,rgb_id=r['sample_id'],kind='natural') for r in rr if r['eligible']]
    augmented=[]
    from clickagent.public_evaluation import point_in_geometry
    for c in cases:
        p=extra_path(c)
        if not p.exists():continue
        a,m=run.load(p)
        def hit(point):return point is not None and point_in_geometry([point[0]/c['image_size'][0],point[1]/c['image_size'][1]],labels[c['sample_id']])
        assert not hit(m['base']) and hit(m['alternative'])
        augmented.append(dict(m,**a,delta=1,kind='mined_positive'))
    assert len(augmented)==20
    all_records=natural+augmented;folds={}
    for source in sorted({c['source'] for c in cases}):
        groups=sorted({c['source_group'] for c in cases if c['source']==source},key=lambda g:run.key(str(g)+'|v31_cv'))
        folds.update({(source,str(g)):j%3 for j,g in enumerate(groups)})
    cv=[];audits=[]
    for fold in range(3):
        fitting=[r for r in all_records if folds[r['source'],str(r['source_group'])]!=fold]
        val=[r for r in all_records if folds[r['source'],str(r['source_group'])]==fold]
        assert not {r['rgb_id'] for r in fitting}&{r['rgb_id'] for r in val}
        audits.append(dict(fold=fold,fit_rgb=sorted({r['rgb_id'] for r in fitting}),validation_rgb=sorted({r['rgb_id'] for r in val})))
        for mode in MODES:
            model=fit_rgb(fitting,mode);pred=scores(model,val)
            cv.extend(dict(sample_id=r['sample_id'],kind=r['kind'],mode=mode,delta=r['delta'],utility=float(p),switch=bool(p>.1)) for r,p in zip(val,pred))
    models={mode:run.save_model(OUT/'models'/f'{mode}.npz',fit_rgb(all_records,mode)) for mode in MODES}
    run.write(OUT/'training_crossfit.json',dict(rows=cv,splits=audits,natural_pairs=len(natural),added_pairs=20,training_rgb=len({r['rgb_id'] for r in all_records}),total_pairs=len(all_records)))
    run.write(OUT/'frozen.json',dict(complete=True,frozen_at_utc=run.now(),development_used=False,models=models,
        training_ids=[c['sample_id'] for c in cases],training_input_hashes={str(p):run.sha(p) for p in [*[OUT/'features'/(run.key(c['sample_id'])+'.npz') for c in cases],*list((OUT/'augmented_features').glob('*.npz'))]}))

def analyze():
    rows=run.read(OUT/'predictions.json')['rows'];base={r['sample_id']:r for r in run.read(CONTEXT/'predictions.json')['rows']};results={};lines=[]
    for cohort in ['all','new_development','old_val']:
        rr=[r for r in rows if cohort=='all' or r['cohort']==cohort];w=run.bootstrap_weights(rr)
        methods={name:100*float(np.mean([r['hits'][name] for r in rr])) for name in rr[0]['hits']};comparisons={}
        for tag,b in [('aug-context','context'),('aug-blurred','blurred'),('aug-base','short_single')]:
            diff=np.asarray([int(r['hits']['combined'])-int(base[r['sample_id']]['hits']['combined'] if b=='context' else r['hits'][b]) for r in rr])
            level=1-.05/3 if cohort=='all' else .95;alpha=(1-level)/2;ci=(100*np.quantile(w@diff,[alpha,1-alpha])).tolist()
            comparisons[tag]=dict(delta_pp=100*float(diff.mean()),ci_pp=ci,positive=bool(ci[0]>0))
        replacements={mode:dict(switches=sum(r['methods'][mode]!=r['methods']['short_single'] for r in rr),repairs=sum(r['switches'][mode]['repair'] for r in rr),harms=sum(r['switches'][mode]['harm'] for r in rr)) for mode in MODES}
        results[cohort]=dict(n=len(rr),percent=methods,comparisons=comparisons,replacements=replacements);lines += [cohort,str(methods),str(comparisons),str(replacements)]
    cv=run.read(OUT/'training_crossfit.json');summary={kind:{mode:dict(n=sum(r['kind']==kind and r['mode']==mode for r in cv['rows']),net_repairs=sum(r['delta'] for r in cv['rows'] if r['kind']==kind and r['mode']==mode and r['switch'])) for mode in MODES} for kind in ['natural','mined_positive']}
    results.update(complete=True,training_crossfit=summary,training_rgb=cv['training_rgb'],training_pairs=cv['total_pairs'],added_pairs=20,new_training_rgb=0)
    run.write(OUT/'results.json',results);lines += ['TrainingCV:'+str(summary),'No independent images added;20existingRGB gained valid positive alternatives. Natural DEV pairs never changed.']
    (OUT/'readout.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    for split in cv['splits']:assert not set(split['fit_rgb'])&set(split['validation_rgb'])
    for p,h in run.read(OUT/'frozen.json')['training_input_hashes'].items():assert run.sha(p)==h
    run.write(OUT/'verification.json',dict(complete=True,at_utc=run.now(),same_rgb_weight_control=True,group_splits=3,added_pairs=20,new_training_rgb=0,
        mainline_unchanged=True,primary_gates={k:v['positive'] for k,v in results['all']['comparisons'].items()}))
    run.write(OUT/'completion.json',dict(complete=True,at_utc=run.now(),results_sha256=run.sha(OUT/'results.json'),verification_sha256=run.sha(OUT/'verification.json')))
    run.status('complete',positive_supplement=True)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','all']);args=parser.parse_args()
    if args.stage=='prepare':prepare();return
    verify();assert run.read(CONTEXT/'completion.json')['complete'];run.OUT=OUT
    cases=run.read(OUT/'data/public.json');train_cases=[c for c in cases if c['cohort']=='train'];dev=[c for c in cases if c['cohort']!='train']
    for c in cases:
        p=OUT/'features'/(run.key(c['sample_id'])+'.npz');p.parent.mkdir(parents=True,exist_ok=True)
        if not p.exists():shutil.copyfile(CONTEXT/'features'/p.name,p)
    rt=run.Runtime();lookup={c['sample_id']:c for c in train_cases}
    for i,row in enumerate(run.read(OUT/'mining.json')['rows'],1):
        c=lookup[row['sample_id']]
        if not extra_path(c).exists():collect_extra(rt,c,row)
        run.status('mined_training_observations',completed=i,total=20)
    if not (OUT/'frozen.json').exists():train(train_cases)
    run.evaluate(dev);analyze()

if __name__=='__main__':main()
