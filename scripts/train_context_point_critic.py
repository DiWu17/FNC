"""Train point utility on old train images; old validation alone selects epochs."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
import shapely
from clickagent.action_benchmark import sha256,write_json
from clickagent.point_critic import FrozenPointEncoder,PointUtilityCritic,point_utility_loss
from clickagent.spatial_geometry import shape_from_geometry
from point_aligned_data import archive_record,read
from point_aligned_predict import load_predictors,predict_record
from run_target_token_experiment import key,read_jsonl

OUT=Path('runs/context_verifier_v9')
SOURCES=['src/clickagent/point_critic.py','scripts/train_context_point_critic.py']


def raw_path(case):
    name=key(case['sample_id'])+'.npz'
    path=Path('runs/semantic_head_v7/features')/name
    return path if path.exists() else Path('runs/conditioned_v3/features')/name


def load_arrays(path):
    with np.load(path,allow_pickle=False) as z:
        return {k:z[k].copy() for k in ('target_attention','target_hidden','image_hidden','vision_hidden')}


def training_points(case,prediction,others):
    rng=np.random.default_rng(int(key(case['sample_id'])[:8],16));w,h=case['image_size']
    order=np.argsort(-np.asarray(prediction['all_scores']),kind='stable')[:64]
    points=[prediction['all_points'][i] for i in order]
    shape=shape_from_geometry(case['target_geometry']);bounds=shape.bounds
    # Annotation-derived coordinates are TRAINING AUGMENTATION ONLY.
    inside=[]
    for _ in range(500):
        xy=rng.uniform(bounds[:2],bounds[2:])
        if shape.covers(shapely.Point(xy)):
            inside.append(np.rint(xy*np.asarray([w,h])).astype(int).tolist())
            if len(inside)==6:break
    points+=inside
    x0,y0,x1,y1=bounds
    for delta in [2,6,16]:
        points.extend([[round(x0*w-delta),round((y0+y1)*h/2)],
            [round(x1*w+delta),round((y0+y1)*h/2)],
            [round((x0+x1)*w/2),round(y0*h-delta)],
            [round((x0+x1)*w/2),round(y1*h+delta)]])
    for other in others:
        b=shape_from_geometry(other['target_geometry']).bounds
        points.append([round((b[0]+b[2])*w/2),round((b[1]+b[3])*h/2)])
    points+=rng.integers([0,0],[w,h],size=(8,2)).tolist()
    return np.unique(np.clip(np.asarray(points),[0,0],[w-1,h-1]),axis=0)


def prepare():
    models=load_predictors(primary_only=True);encoder=FrozenPointEncoder(models['joint_connected_s1'][0])
    cases=read_jsonl(OUT/'manifest.jsonl');train=[c for c in cases if c['experiment_split']=='train']
    byimage={}
    for c in train:byimage.setdefault(c['pixel_sha256'],[]).append(c)
    for split in ('train','val'):
        subset=[c for c in cases if c['experiment_split']==split];dest=OUT/'critic_cache'/split;dest.mkdir(parents=True,exist_ok=True)
        for i,case in enumerate(subset,1):
            path=dest/(key(case['sample_id'])+'.npz')
            if path.exists():continue
            raw=raw_path(case);record=archive_record(raw,case,labels=False)
            if not record['metadata']['target_available']:
                np.savez_compressed(path,available=False,sample_id=case['sample_id']);continue
            pred=predict_record(record,models,only=['joint_connected_s1'],top_k=5)['joint_connected_s1']
            points=training_points(case,pred,byimage[case['pixel_sha256']]) if split=='train' else np.asarray([v['point'] for v in pred['candidates'][:4]])
            arrays=load_arrays(raw);state=encoder.prepare(arrays,record['metadata']['grid_shape'],case['image_size'])
            features=encoder.features(state,points)
            labels=shapely.covers(shape_from_geometry(case['target_geometry']),shapely.points(points/np.asarray(case['image_size'])))
            np.savez_compressed(path,available=True,sample_id=case['sample_id'],features=features.astype(np.float16),
                labels=np.asarray(labels,np.float32),points=points,raw_sha256=sha256(raw),
                annotation_derived_points=split=='train')
            if i%25==0:print(json.dumps(dict(stage='critic_cache',split=split,i=i,n=len(subset))),flush=True)
        write_json(OUT/f'critic_cache_{split}.json',dict(complete=True,n=len(subset),
            cases_sha256=sha256(OUT/'manifest.jsonl'),sources={p:sha256(Path(p)) for p in SOURCES}))


def dataset(split):
    result=[]
    for path in sorted((OUT/'critic_cache'/split).glob('*.npz')):
        with np.load(path,allow_pickle=False) as z:
            if bool(z['available']):result.append((torch.from_numpy(z['features'].astype(np.float32)),torch.from_numpy(z['labels'].astype(np.float32)),str(z['sample_id'])))
    return result


@torch.inference_mode()
def evaluate(model,data):
    model.eval();hits=0;oracle=0;losses=[]
    for x,y,_ in data:
        scores=model(x);hits+=int(y[scores.argmax()]>0);oracle+=int(y.any())
        losses.append(float(point_utility_loss(scores,y,[len(y)])))
    return dict(named=len(data),hits=hits,point_oracle=oracle,loss=float(np.mean(losses)))


def train():
    assert not list((OUT/'features').glob('*.npz')),'Freeze before fresh test extraction'
    train=dataset('train');val=dataset('val');joined=torch.cat([v[0] for v in train]);mean=joined.mean(0);std=joined.std(0);del joined
    (OUT/'models').mkdir(exist_ok=True)
    for suffix,seed in [('s1',20261122),('s2',20261123)]:
        dest=OUT/'models'/f'point_critic_{suffix}.pt';history_path=dest.with_suffix('.json')
        if history_path.exists():continue
        torch.manual_seed(seed);model=PointUtilityCritic(mean,std);optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=.01)
        best=None;history=[];started=time.perf_counter()
        for epoch in range(26):
            losses=[]
            if epoch:
                model.train();order=np.random.default_rng(seed+epoch).permutation(len(train))
                for start in range(0,len(order),12):
                    batch=[train[int(i)] for i in order[start:start+12]]
                    x=torch.cat([v[0] for v in batch]);y=torch.cat([v[1] for v in batch]);lengths=[len(v[1]) for v in batch]
                    optimizer.zero_grad(set_to_none=True);loss=point_utility_loss(model(x),y,lengths)
                    loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();losses.append(float(loss.detach()))
            result=evaluate(model,val);row=dict(epoch=epoch,validation=result,train_loss=float(np.mean(losses)) if losses else None,
                elapsed_seconds=time.perf_counter()-started);history.append(row)
            selector=(result['hits'],-result['loss'])
            if best is None or selector>best:
                best=selector;selected=row
                torch.save(dict(state_dict=model.state_dict(),width=model.width,metadata=dict(seed=seed,selected=row,
                    train_queries=len(train),validation_queries=len(val),backbone_frozen=True,
                    frozen_encoder='v8 joint_connected_s1',same_global_features_for_all_candidate_sources=True)),dest)
            print(json.dumps(dict(stage='critic_train',seed=suffix,**row)),flush=True)
        write_json(history_path,dict(complete=True,selected=selected,history=history,sha256=sha256(dest),
            parameters=sum(p.numel() for p in model.parameters()),sources={p:sha256(Path(p)) for p in SOURCES}))


def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['cache','train'],required=True);args=p.parse_args()
    torch.set_num_threads(4)
    prepare() if args.stage=='cache' else train()


if __name__=='__main__':main()
