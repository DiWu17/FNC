"""Fit target-conditioned action readouts on official training images only."""
import argparse
from collections import Counter,defaultdict
import gc
import json
from pathlib import Path
import random
import time

import numpy as np
import torch

from clickagent.action_benchmark import sha256,write_json
from clickagent.action_grounding import AttentionPointGrounder
from clickagent.conditioned_grounder import (ConditionedSpatialGrounder,SpatialConfig,make_batch,
    save_spatial_checkpoint,load_spatial_checkpoint,spatial_loss)
from clickagent.public_evaluation import point_in_geometry
from clickagent.spatial_geometry import spatial_supervision,decode_candidates
from run_target_token_experiment import read_json,read_jsonl,key


VARIANTS=('full','attention_only','no_image','no_mass','no_query','no_attention','static_96')


def load_records(out,cases,with_labels=True):
    records=[];images={};skipped=[]
    for i,c in enumerate(cases,1):
        with np.load(out/'features'/f"{key(c['sample_id'])}.npz",allow_pickle=False) as z:
            meta=json.loads(str(z['metadata']))
            r={name:z[name].copy() for name in ('target_attention','target_hidden','image_hidden','vision_hidden')}
        assert meta['sample_id']==c['sample_id'] and meta['image_sha256']==c['image_sha256']
        assert meta['own_token_forward'] and meta['ground_truth_in_model_input'] is False
        r.update(case=c,metadata=meta)
        if not meta['target_available']:
            skipped.append(c['sample_id']);records.append(r);continue
        for field in ('image_hidden','vision_hidden'):
            image=images.get((c['pixel_sha256'],field))
            if image is not None and np.array_equal(image,r[field]):r[field]=image
            else:images[c['pixel_sha256'],field]=r[field]
        if with_labels:r['label_weights'],r['label_offsets']=spatial_supervision(c['target_geometry'],meta['fine_grid_shape'])
        records.append(r)
        if i%100==0:print(f'Cached {i}/{len(cases)} train/val samples',flush=True)
    return records,skipped


def batches(records,batch_size,rng=None):
    groups=defaultdict(list)
    for r in records:
        if r['metadata']['target_available']:groups[tuple(r['metadata']['grid_shape'])].append(r)
    result=[]
    for grid,group in sorted(groups.items()):
        if rng:rng.shuffle(group)
        result.extend(group[i:i+batch_size] for i in range(0,len(group),batch_size))
    if rng:rng.shuffle(result)
    return result


@torch.inference_mode()
def evaluate(model,records,device,batch_size):
    model.eval();hits=0;center_hits=0;losses=[];dataset_hits=Counter()
    for group in batches(records,batch_size):
        inputs,target=make_batch(group,device);result=model(**inputs)
        loss,_=spatial_loss(result,**target);losses.extend([float(loss)]*len(group))
        logits=result['logits'].cpu().numpy();offsets=result['offsets'].cpu().numpy()
        for i,r in enumerate(group):
            c=r['case'];meta=r['metadata'];w,h=c['image_size']
            for center in (False,True):
                point=decode_candidates(logits[i],offsets[i],result['grid_shape'],meta['image_size'],top_k=1,center_only=center)[0]['point']
                hit=point_in_geometry([point[0]/w,point[1]/h],c['target_geometry'])
                if center:center_hits+=hit
                else:hits+=hit;dataset_hits[c['dataset']]+=hit
    return {'n':len(records),'hits':int(hits),'accuracy':hits/len(records),'center_hits':int(center_hits),
        'named_samples':sum(r['metadata']['target_available'] for r in records),
        'loss_on_named':float(np.mean(losses)) if losses else None,
        'datasets':{ds:{'n':sum(r['case']['dataset']==ds for r in records),'hits':int(dataset_hits[ds])}
            for ds in sorted({r['case']['dataset'] for r in records})}}


def train_variant(out,variant,records,config,device,epochs,batch_size,lr,initial,seed):
    path=out/'models'/f'{variant}.pt';history_path=out/'models'/f'{variant}_history.json'
    # A completed variant is immutable during resumption of the same experiment.
    if history_path.exists():
        old=read_json(history_path)
        assert old['complete'] and sha256(path)==old['checkpoint_sha256']
        return old
    torch.manual_seed(seed);np.random.seed(seed);random.seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    config=SpatialConfig(**dict(config,variant=variant))
    model=ConditionedSpatialGrounder(config,initial).to(device)
    train=[r for r in records if r['case']['experiment_split']=='train' and r['metadata']['target_available']]
    val=[r for r in records if r['case']['experiment_split']=='val']
    optimizer=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=.01)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=epochs,eta_min=lr*.1)
    first=evaluate(model,val,device,batch_size)
    print(f'{variant} epoch 0: '+json.dumps(first),flush=True)
    history=[dict(epoch=0,val=first)];best_key=(first['hits'],-first['loss_on_named']);best_epoch=0
    metadata={'variant':variant,'seed':seed,'best_epoch':0,'validation':first,
        'train_cases':len(train),'backbone_trainable_parameters':0,'head_parameters':sum(p.numel() for p in model.parameters())}
    save_spatial_checkpoint(path,model,metadata)
    rng=random.Random(seed);started=time.perf_counter();updates=0
    for epoch in range(1,epochs+1):
        model.train();epoch_losses=[]
        for group in batches(train,batch_size,rng):
            inputs,target=make_batch(group,device)
            optimizer.zero_grad(set_to_none=True)
            result=model(**inputs);loss,_=spatial_loss(result,**target)
            if not torch.isfinite(loss):raise ValueError('Nonfinite spatial training loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            optimizer.step();updates+=1;epoch_losses.extend([float(loss.detach())]*len(group))
        scheduler.step()
        val_metrics=evaluate(model,val,device,batch_size)
        record={'epoch':epoch,'train_loss':float(np.mean(epoch_losses)),'val':val_metrics,
            'elapsed_seconds':time.perf_counter()-started,'updates':updates}
        history.append(record);print(variant+' '+json.dumps(record),flush=True)
        rank=(val_metrics['hits'],-val_metrics['loss_on_named'])
        if rank>best_key:
            best_key=rank;best_epoch=epoch
            metadata=dict(metadata,best_epoch=epoch,validation=val_metrics,training_updates=updates)
            save_spatial_checkpoint(path,model,metadata)
        write_json(out/'models'/f'{variant}_progress.json',{'history':history,'best_epoch':best_epoch})
        if epoch>=16 and epoch-best_epoch>=10:break
    selected_model,_=load_spatial_checkpoint(path,device)
    train_metrics=evaluate(selected_model,[r for r in records if r['case']['experiment_split']=='train'],device,batch_size)
    print(variant+' selected-checkpoint train metrics: '+json.dumps(train_metrics),flush=True)
    del selected_model
    result={'complete':True,'variant':variant,'history':history,'best_epoch':best_epoch,
        'train_metrics_at_selected_checkpoint':train_metrics,
        'best_validation':history[best_epoch]['val'],'checkpoint_sha256':sha256(path),'head_parameters':metadata['head_parameters'],
        'epochs_executed':epoch,'training_updates':updates,'config':config.__dict__}
    write_json(history_path,result)
    del model,optimizer;gc.collect()
    if torch.cuda.is_available():torch.cuda.empty_cache()
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=Path('runs/conditioned_v3'))
    p.add_argument('--epochs',type=int,default=40);p.add_argument('--batch-size',type=int,default=8)
    p.add_argument('--lr',type=float,default=3e-4);p.add_argument('--variants',nargs='+',choices=VARIANTS,default=list(VARIANTS))
    p.add_argument('--device',default='cuda:0');args=p.parse_args();out=args.output
    if 'full' not in args.variants:raise ValueError('The fixed primary full model must be trained')
    assert read_json(out/'features_trainval_completion.json')['complete']
    if (out/'frozen_readouts.json').exists():raise ValueError('Readouts already frozen; do not retrain after test')
    cases=read_jsonl(out/'manifest.jsonl');test=[c for c in cases if c['experiment_split']=='test']
    assert not any((out/'features'/f"{key(c['sample_id'])}.npz").exists() for c in test)
    used=[c for c in cases if c['experiment_split'] in ('train','val')]
    assert all(c['official_training_split'] and c['source_split']=='train' for c in used)
    torch.set_num_threads(4);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    (out/'models').mkdir(exist_ok=True)
    sources=['src/clickagent/'+x for x in ('conditioned_grounder.py','spatial_geometry.py','public_evaluation.py')]+[str(Path(__file__))]
    training_identity={'manifest_sha256':sha256(out/'manifest.jsonl'),'feature_identity_sha256':sha256(out/'feature_identity.json'),
        'source_sha256':{name:sha256(Path(name)) for name in sources},'epochs':args.epochs,'batch_size':args.batch_size,'lr':args.lr,
        'variants':list(args.variants),'seed':20261101,'initializer_sha256':sha256(Path('runs/action_v3/grounder/grounder.joblib')),
        'protocol_sha256':sha256(Path('docs/CONDITIONED_SPATIAL_PROTOCOL.md'))}
    if (out/'training_identity.json').exists() and read_json(out/'training_identity.json')!=training_identity:
        raise ValueError('Training source/parameters changed during resume')
    write_json(out/'training_identity.json',training_identity)
    records,missing=load_records(out,used)
    first=next(r for r in records if r['metadata']['target_available'])
    config={'image_dim':first['image_hidden'].shape[-1],'vision_dim':first['vision_hidden'].shape[-1],'query_dim':first['target_hidden'].shape[-1],
        'heads':first['target_attention'].shape[1],'query_layers':3,'width':128}
    old=AttentionPointGrounder.load('runs/action_v3/grounder/grounder.joblib').pipeline
    initial=(old.named_steps['classifier'].coef_[0]/old.named_steps['scale'].scale_*.25).astype(np.float32)
    histories={name:train_variant(out,name,records,config,args.device,args.epochs,args.batch_size,args.lr,initial,20261101) for name in args.variants}
    assert not any((out/'features'/f"{key(c['sample_id'])}.npz").exists() for c in test)
    frozen={'primary_model':'full','models':{name:{'file':name+'.pt','sha256':r['checkpoint_sha256'],
        'best_epoch':r['best_epoch'],'validation':r['best_validation'],'head_parameters':r['head_parameters']} for name,r in histories.items()},
        'head_sources':training_identity['source_sha256'],'training_identity_sha256':sha256(out/'training_identity.json'),
        'test_features_at_freeze':0,'test_labels_used_for_fitting':0,'test_n':len(test),
        'missing_name_trainval_samples':missing,'valid_train_images':len({r['case']['pixel_sha256'] for r in records if r['metadata']['target_available'] and r['case']['experiment_split']=='train'}),
        'note':'Primary full architecture fixed in protocol; each checkpoint chosen only by validation point hits, then validation loss. All test failures remain in denominator.'}
    write_json(out/'frozen_readouts.json',frozen)
    print('FROZEN '+json.dumps(frozen,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
