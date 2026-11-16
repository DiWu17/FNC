"""Adapt 97-coefficient spatial readouts on dev images only, before test scores.

Prefill and generated-target first-token readouts have identical supervision,
features, model family and cross-validation. The original frozen-head timing
comparison is preserved separately, never replaced by these adapted results.
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from clickagent.action_benchmark import sha256,write_json
from clickagent.action_grounding import AttentionPointGrounder,patch_features,_pixel_centers
from clickagent.public_evaluation import point_in_geometry
from clickagent.semantic_target import temporal_map
from run_semantic_target import load_feature,score_map
from run_target_token_experiment import read_json,read_jsonl,key


CS=(.001,.01,.1,1.)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('runs/semantic_target_v2'))
    args=parser.parse_args();output=args.output
    dest=output/'adapted_readouts';dest.mkdir(exist_ok=True)
    if (output/'predictions.jsonl').exists():
        raise ValueError('Test scores already exist; use a new untouched test for fitting a new branch')
    choice=read_json(output/'frozen_selection.json')['selected']
    # Filter before opening any feature cache or using any geometric labels.
    cases=[c for c in read_jsonl(output/'manifest.jsonl') if c['experiment_split']=='dev']
    assert len(cases)==64
    folds={}
    for ds in sorted({c['dataset'] for c in cases}):
        group=sorted([c for c in cases if c['dataset']==ds],key=lambda c:key('readout:'+c['sample_id']))
        for i,c in enumerate(group):folds[c['sample_id']]=i%4
    report={'fit_sample_ids':[c['sample_id'] for c in cases],'n_dev':64,'test_features_loaded':0,
        'test_predictions_exist_at_start':False,'temporal_selection':choice,
        'fit_source_sha256':sha256(Path(__file__)),'heads':{},
        'note':'Only dev images fit external 96-feature logistic readouts. Same 4-fold image-level CV for both stages; point centers and backbone unchanged.'}
    for stage in ('prefill','target'):
        records=[]
        for c in cases:
            raw,meta=load_feature(output,c,choice['variant'])
            attention=raw[0] if stage=='prefill' else temporal_map(raw,meta,choice['policy'])
            grid=meta['grid_shape'];w,h=c['image_size']
            x,y=_pixel_centers(grid,c['image_size'])
            labels=np.asarray([point_in_geometry([float(a)/w,float(b)/h],c['target_geometry']) for a,b in zip(x,y)],int)
            records.append((c,meta,None if attention is None else patch_features(attention,grid),labels,attention))
        results=[]
        for C in CS:
            hits=0
            for fold in range(4):
                train=[r for r in records if folds[r[0]['sample_id']]!=fold and r[2] is not None and r[3].any()]
                pipe=Pipeline([('scale',StandardScaler()),('classifier',LogisticRegression(C=C,class_weight='balanced',solver='lbfgs',max_iter=2000,random_state=20261031))])
                pipe.fit(np.concatenate([r[2] for r in train]),np.concatenate([r[3] for r in train]))
                model=AttentionPointGrounder(pipe,96,C,tuple(records[0][1]['head_order']))
                for c,meta,features,labels,attention in records:
                    if folds[c['sample_id']]!=fold:continue
                    hits+=score_map(attention,meta,c,model)['hit']
            results.append({'C':C,'cross_validated_hits':int(hits),'n':64})
            print(stage+' CV '+json.dumps(results[-1]),flush=True)
        best=max(results,key=lambda r:(r['cross_validated_hits'],-r['C']))
        train=[r for r in records if r[2] is not None and r[3].any()]
        pipe=Pipeline([('scale',StandardScaler()),('classifier',LogisticRegression(C=best['C'],class_weight='balanced',solver='lbfgs',max_iter=2000,random_state=20261031))])
        pipe.fit(np.concatenate([r[2] for r in train]),np.concatenate([r[3] for r in train]))
        model=AttentionPointGrounder(pipe,96,best['C'],tuple(records[0][1]['head_order']))
        joblib.dump(model,dest/f'{stage}.joblib')
        report['heads'][stage]={'selected':best,'search':results,'fit_images_used':len(train),
            'no_positive_center_or_name_images_skipped_for_fit':len(records)-len(train),
            'coefficient_count':int(pipe.named_steps['classifier'].coef_.size+pipe.named_steps['classifier'].intercept_.size),
            'sha256':sha256(dest/f'{stage}.joblib')}
    assert not (output/'predictions.jsonl').exists(),'Test scores appeared before head freeze; audit this run'
    report['test_predictions_exist_at_freeze']=False
    write_json(dest/'frozen_weights.json',report)
    print('FROZEN adapted heads '+json.dumps(report['heads']),flush=True)


if __name__=='__main__':main()
