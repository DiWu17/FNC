"""Four matched warm-start controls, two seeds; only old validation selects."""
import argparse
import gc
import json
from pathlib import Path
import random
import time

import numpy as np
import torch

from clickagent.action_benchmark import sha256,write_json
from clickagent.point_aligned_readout import (aligned_loss,all_candidate_points,candidate_point_hits,
    load_warm_start,save_aligned,load_aligned)
from point_aligned_data import OUT,BASE,read,batch,training_records,frozen_source_identity


SOURCES=['src/clickagent/point_aligned_readout.py','scripts/point_aligned_data.py',
         'scripts/train_point_aligned.py','src/clickagent/semantic_head_ranker.py',
         'src/clickagent/candidate_ranker.py','src/clickagent/attention_field_ranker.py']


def selected_metrics(output,group,decoder):
    points=[r['frozen_points'] for r in group] if decoder=='fixed' else all_candidate_points(output,group)
    hits=[r['label_fixed_point_hit'].astype(bool) for r in group] if decoder=='fixed' else candidate_point_hits(points,group)
    scores=output['scores'].detach().cpu().numpy(); cursor=0; result=[]
    for r,pp,hh in zip(group,points,hits):
        n=len(pp);ci=int(scores[cursor:cursor+n].argmax());cursor+=n
        result.append(dict(hit=bool(hh[ci]),oracle=bool(hh.any()),region=bool(r['label_area'][ci].sum()>0)))
    return result


@torch.inference_mode()
def evaluate(model,records,job):
    model.eval(); model.cpu(); metrics=[]; losses=[]
    for start in range(0,len(records),8):
        group=[r for r in records[start:start+8] if r['metadata']['target_available']]
        if not group:continue
        inputs,target=batch(group,'cpu'); output=model(**inputs)
        point_labels=target['label_fixed_point_hit'] if job['decoder']=='fixed' else torch.from_numpy(np.concatenate(candidate_point_hits(all_candidate_points(output,group),group)))
        loss,_=aligned_loss(output,target,point_labels,job['objective']=='point')
        losses.extend([float(loss)]*len(group));metrics+=selected_metrics(output,group,job['decoder'])
    return dict(n=len(records),named=len(metrics),hits=sum(r['hit'] for r in metrics),
        regions=sum(r['region'] for r in metrics),point_oracle=sum(r['oracle'] for r in metrics),loss=float(np.mean(losses)))


def train_one(job,train,val,protocol,device):
    name=job['name'];history_path=OUT/'models'/f'{name}_history.json'
    if history_path.exists():
        history=read(history_path);assert history['complete'];assert history['protocol_sha256']==sha256(OUT/'protocol.json')
        assert sha256(OUT/'models'/f'{name}.pt')==history['checkpoint_sha256'];return
    seed=job['seed'];torch.manual_seed(seed);np.random.seed(seed);random.seed(seed)
    warm=BASE/'models'/f'semantic_hard_{job["suffix"]}__point.pt'
    assert sha256(warm)==protocol['warm_start_sha256'][str(warm)]
    model=load_warm_start(warm,job['connect_fine'],device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=protocol['learning_rate'],weight_decay=.01)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,protocol['epochs'],eta_min=protocol['learning_rate']*.1)
    named=[r for r in train if r['metadata']['target_available']];history=[];best=None;checkpoint=None
    started=time.perf_counter()
    for epoch in range(protocol['epochs']+1):
        losses=[];steps=0
        if epoch:
            model.to(device).train()
            order=np.random.default_rng(np.random.SeedSequence([seed,epoch,8])).permutation(len(named))[:600]
            for start in range(0,600,8):
                group=[named[int(i)] for i in order[start:start+8]]
                inputs,target=batch(group,device);optimizer.zero_grad(set_to_none=True);output=model(**inputs)
                point_labels=target['label_fixed_point_hit'] if job['decoder']=='fixed' else torch.from_numpy(np.concatenate(candidate_point_hits(all_candidate_points(output,group),group))).to(device)
                loss,_=aligned_loss(output,target,point_labels,job['objective']=='point')
                assert torch.isfinite(loss)
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
                losses.append(float(loss.detach()));steps+=1
            scheduler.step();assert steps==75
        metrics=evaluate(model,val,job)
        row=dict(epoch=epoch,validation=metrics,train_loss=float(np.mean(losses)) if losses else None,
            consumed_queries=600 if epoch else 0,optimizer_steps=steps,elapsed_seconds=time.perf_counter()-started)
        history.append(row)
        selector=(metrics['hits'],metrics['regions'],-metrics['loss'])
        if best is None or selector>best:
            best=selector;checkpoint=dict(epoch=epoch,validation=metrics)
            save_aligned(OUT/'models'/f'{name}.pt',model,dict(job=job,**checkpoint,protocol_sha256=sha256(OUT/'protocol.json')))
        print(json.dumps(dict(job=name,**row),ensure_ascii=False),flush=True)
    write_json(history_path,dict(complete=True,job=job,history=history,selected=checkpoint,
        checkpoint_sha256=sha256(OUT/'models'/f'{name}.pt'),protocol_sha256=sha256(OUT/'protocol.json'),
        parameters=sum(p.numel() for p in model.parameters())))
    del model,optimizer;gc.collect();torch.cuda.empty_cache()


def freeze(protocol):
    if (OUT/'frozen_readouts.json').exists():
        frozen=read(OUT/'frozen_readouts.json');assert frozen['protocol_sha256']==sha256(OUT/'protocol.json');return
    assert not list((OUT/'features').glob('*.npz')) if (OUT/'features').exists() else True
    sources={p:sha256(Path(p)) for p in SOURCES}
    assert sources==protocol['training_sources_sha256']
    models={}
    for job in protocol['jobs']:
        h=read(OUT/'models'/f'{job["name"]}_history.json');assert h['complete']
        assert len(h['history'])==protocol['epochs']+1
        assert [r['optimizer_steps'] for r in h['history']]==[0]+[75]*protocol['epochs']
        assert sha256(OUT/'models'/f'{job["name"]}.pt')==h['checkpoint_sha256']
        best=max(h['history'],key=lambda r:(r['validation']['hits'],r['validation']['regions'],-r['validation']['loss']))
        assert best['epoch']==h['selected']['epoch']
        models[job['name']]=dict(file=f'{job["name"]}.pt',sha256=h['checkpoint_sha256'],job=job,selected=h['selected'])
    write_json(OUT/'frozen_readouts.json',dict(complete=True,models=models,sources=sources,
        protocol_sha256=sha256(OUT/'protocol.json'),manifest_sha256=sha256(OUT/'manifest.jsonl'),
        primary='joint_connected_s1',fresh_test_features_at_freeze=0,backbone_trainable_parameters=0,
        frozen_at=time.time()))
    print('ALL EIGHT READOUTS FROZEN',flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--suffix',choices=['s1','s2']);p.add_argument('--device',default='cuda:0');p.add_argument('--freeze',action='store_true')
    args=p.parse_args();torch.set_num_threads(4);protocol=read(OUT/'protocol.json')
    for path,digest in protocol['training_sources_sha256'].items():assert sha256(Path(path))==digest,path
    if args.freeze:return freeze(protocol)
    assert args.suffix
    train=training_records('train');val=training_records('val')
    for job in protocol['jobs']:
        if job['suffix']==args.suffix:train_one(job,train,val,protocol,args.device)


if __name__=='__main__':main()
