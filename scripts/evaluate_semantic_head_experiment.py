"""Score all prespecified v7 checkpoints on diagnostic and untouched cohorts."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np
import torch

from clickagent.action_benchmark import sha256, write_json
from clickagent.attention_field_ranker import load_field_ranker, make_field_batch
from clickagent.candidate_ranker import decode_ranker, load_ranker, make_ranker_batch
from clickagent.conditioned_grounder import make_batch as make_raw_batch
from clickagent.phrase_grounder import load_phrase, decode_phrase
from clickagent.semantic_head_experiment import load_readout, score_output
from evaluate_conditioned_readouts import score_point
from evaluate_phrase_readouts import paired_interval
from run_target_token_experiment import key, read_jsonl
from semantic_head_data import load_records, make_batch


def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))


def prediction(case, method, point, **extra):
    return dict(sample_id=case['sample_id'], dataset=case['dataset'], method=method, point=point,
        hit=bool(score_point(point, case)), **extra)


@torch.inference_mode()
def score_checkpoint(name, model, records, rule, device):
    named=[r for r in records if r['metadata']['target_available']]; rows=[]
    modes=['rank'] if rule=='region' else ['combined'] if rule=='quality' else ['rank','combined','quality']
    for start in range(0,len(named),8):
        group=named[start:start+8]; inputs,_=make_batch(group,device,labels=False); output=model(**inputs)
        cpu={k:v.cpu() if isinstance(v,torch.Tensor) else v for k,v in output.items()}; cursor=0
        for i,r in enumerate(group):
            c=r['case']; positive=r['label_area'].sum(-1)>0; n=len(r['parent_indices'])
            for mode in modes:
                scored=score_output(cpu,mode); raw_candidates=decode_ranker(scored,r,i,top_k=5); first=raw_candidates[0]
                candidates=[dict(p, own_fine_point=p['point'], point=r['frozen_points'][p['candidate_index']].tolist()) for p in raw_candidates]
                ci=first['candidate_index']; point=r['frozen_points'][ci].tolist()
                center=decode_ranker(scored,r,i,mode='center',top_k=1)[0]['point']
                hit=bool(score_point(point,c)); region=bool(positive[ci]); covered=bool(positive.any())
                stage='hit' if hit else 'fixed_fine_miss_after_intersecting_parent' if region else 'ranking_miss_with_target_candidate' if covered else 'no_target_peak_candidate'
                extras=dict(candidates=candidates,selected_candidate_index=ci,selected_parent_index=first['parent_index'],
                    selected_region_overlap=region,candidate_region_oracle=covered,
                    frozen_fine_point_oracle=bool(r['label_fixed_point_hit'].any()),
                    region_top3=bool(any(positive[p['candidate_index']] for p in candidates[:3])),
                    region_top5=bool(any(positive[p['candidate_index']] for p in candidates[:5])),
                    predicted_point_quality=float(cpu['point_quality_logits'][cursor+ci].sigmoid()),
                    target_name=r['metadata']['target_text'],failure_stage=stage,n_candidates=n,
                    score_mode=mode,selector=rule,decoder='same frozen v5 fine decoder')
                method=name+'__'+mode
                rows.append(prediction(c,method,point,**extras))
                rows.append(prediction(c,method+'_ownfine',first['point']))
                rows.append(prediction(c,method+'_center',center))
            cursor+=n
    for r in records:
        if not r['metadata']['target_available']:
            for mode in modes:
                for suffix in ['', '_ownfine','_center']:
                    rows.append(prediction(r['case'],name+'__'+mode+suffix,None,failure_stage='no_name'))
    return rows


@torch.inference_mode()
def baselines(out,records,split,device):
    rows=[]; named=[r for r in records if r['metadata']['target_available']]
    checkpoints=[('v5_fixed',Path('runs/candidate_ranker_v5/models/full.pt'),'v5'),
        ('v6_s1_fixed',Path('runs/attention_field_v6/models/all.pt'),'v6'),
        ('v6_s2_fixed',Path('runs/attention_field_v6/models/all_replica.pt'),'v6')]
    protocol=read(out/'protocol.json')
    for name,path,kind in checkpoints:
        assert sha256(path)==protocol['baseline_checkpoint_sha256'][str(path)]
        model,_=load_ranker(path,device) if kind=='v5' else load_field_ranker(path,device)
        for start in range(0,len(named),8):
            group=named[start:start+8]
            inputs,_=(make_ranker_batch if kind=='v5' else make_field_batch)(group,device,labels=False)
            output=model(**inputs); cpu={k:v.cpu() if isinstance(v,torch.Tensor) else v for k,v in output.items()}
            for i,r in enumerate(group):
                candidates=decode_ranker(cpu,r,i,top_k=5);p=candidates[0]; ci=p['candidate_index']; fixed=r['frozen_points'][ci].tolist()
                if kind=='v5': assert fixed==p['point']
                positive=r['label_area'].sum(-1)>0;region=bool(positive[ci]);covered=bool(positive.any());hit=bool(score_point(fixed,r['case']))
                stage='hit' if hit else 'fixed_fine_miss_after_intersecting_parent' if region else 'ranking_miss_with_target_candidate' if covered else 'no_target_peak_candidate'
                rows.append(prediction(r['case'],name,fixed,selected_candidate_index=ci,selected_parent_index=p['parent_index'],
                    selected_region_overlap=region,candidate_region_oracle=covered,
                    frozen_fine_point_oracle=bool(r['label_fixed_point_hit'].any()),failure_stage=stage,
                    region_top3=bool(any(positive[p['candidate_index']] for p in candidates[:3])),
                    region_top5=bool(any(positive[p['candidate_index']] for p in candidates[:5]))))
                if kind=='v6': rows.append(prediction(r['case'],name.replace('_fixed','_ownfine'),p['point']))
        for r in records:
            if not r['metadata']['target_available']:
                rows.append(prediction(r['case'],name,None,failure_stage='no_name'))
                if kind=='v6': rows.append(prediction(r['case'],name.replace('_fixed','_ownfine'),None))
        del model;torch.cuda.empty_cache()
    # Re-run the historical phrase-local decoder on these exact cached trajectories.
    phrase_path=Path('runs/phrase_v4/models/local.pt')
    assert sha256(phrase_path)==protocol['baseline_checkpoint_sha256'][str(phrase_path)]
    model,_=load_phrase(phrase_path,device); grids=defaultdict(list)
    for r in named: grids[tuple(r['metadata']['grid_shape'])].append(r)
    root=out if split=='test' else Path('runs/candidate_ranker_v5')
    for grid,group in sorted(grids.items()):
        for start in range(0,len(group),8):
            raw=[]
            for r in group[start:start+8]:
                with np.load(root/'features'/f'{key(r["case"]["sample_id"])}.npz',allow_pickle=False) as z:
                    arrays={n:z[n].copy() for n in ['target_attention','target_hidden','image_hidden','vision_hidden']}
                raw.append(dict(arrays,case=r['case'],metadata=r['metadata']))
            inputs,_=make_raw_batch(raw,device,labels=False);output=model(**inputs)
            cpu={k:v.cpu() if isinstance(v,torch.Tensor) else v for k,v in output.items()}
            for i,r in enumerate(raw): rows.append(prediction(r['case'],'phrase_local',decode_phrase(cpu,i,r['case']['image_size'],top_k=1)[0]['point']))
    for r in records:
        if not r['metadata']['target_available']: rows.append(prediction(r['case'],'phrase_local',None))
    del model;torch.cuda.empty_cache()
    for r in records:
        c=r['case']; sid=key(c['sample_id'])
        native=read(root/'native'/f'{sid}.json');ui=read(root/'ui_tars'/f'{sid}.json')
        rows.append(prediction(c,'qwen_native_completed_lax',native['completed_lax_point']))
        rows.append(prediction(c,'ui_tars',ui['point']))
    return rows


def planned_comparisons(rows,cases):
    result=[]
    def add(label,a,b):result.append(dict(factor=label,**paired_interval(rows,a,b,cases)))
    for suffix in ['s1','s2']:
        def method(job,selector='point',mode='rank'):return f'{job}_{suffix}__{selector}__{mode}'
        add('point checkpoint selection',method('flat_base'),method('flat_base','region'))
        add('explicit early semantic condition',method('semantic_base'),method('static_base'))
        add('early versus flat architecture',method('semantic_base'),method('flat_base'))
        add('more real multi-target data',method('semantic_expanded'),method('semantic_base'))
        add('same data: grouped hard training',method('semantic_hard'),method('semantic_expanded'))
        add('same checkpoint: point-quality score',method('semantic_hard',mode='combined'),method('semantic_hard'))
        primary=method('semantic_hard','quality','combined')
        add('combined recipe',primary,method('flat_base','region'))
        add('combined versus previous v6',primary,'v6_'+suffix+'_fixed')
        add('combined versus Qwen native',primary,'qwen_native_completed_lax')
        add('combined versus UI-TARS',primary,'ui_tars')
    return result


def evaluate_split(out,split,device):
    protocol=read(out/'protocol.json');frozen=read(out/'frozen_readouts.json')
    for source,digest in frozen['sources'].items():assert sha256(Path(source))==digest,source
    records=load_records(out,split);cases=[r['case'] for r in records];rows=baselines(out,records,split,device)
    for name,entry in frozen['models'].items():
        path=out/'models'/entry['file'];assert sha256(path)==entry['sha256']
        model,_=load_readout(path,device);rows.extend(score_checkpoint(name,model,records,entry['selector'],device))
        print('SCORED '+split+' '+name,flush=True);del model;torch.cuda.empty_cache()
    grouped=defaultdict(list)
    for r in rows:grouped[r['method']].append(r)
    methods={}
    for name,group in sorted(grouped.items()):
        assert len(group)==len(cases)==len({r['sample_id'] for r in group})
        m=dict(n=len(cases),hits=sum(r['hit'] for r in group),points=sum(r['point'] is not None for r in group),
            accuracy=sum(r['hit'] for r in group)/len(cases),datasets={ds:dict(n=sum(r['dataset']==ds for r in group),
                hits=sum(r['hit'] for r in group if r['dataset']==ds)) for ds in sorted({c['dataset'] for c in cases})})
        if any('selected_region_overlap' in r for r in group):
            m.update(region_top1=sum(r.get('selected_region_overlap',False) for r in group),
                region_top3=sum(r.get('region_top3',False) for r in group),region_top5=sum(r.get('region_top5',False) for r in group),
                candidate_region_coverage=sum(r.get('candidate_region_oracle',False) for r in group),
                frozen_fine_point_oracle=sum(r.get('frozen_fine_point_oracle',False) for r in group),
                failure_stages=dict(Counter(r.get('failure_stage') for r in group)))
        methods[name]=m
    dest=out/'evaluation'/split;dest.mkdir(parents=True,exist_ok=True)
    (dest/'predictions.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
    summary=dict(primary=protocol['primary']+'__combined',n=len(cases),split=split,test_is_fresh=split=='test',methods=methods,
        paired_comparisons=planned_comparisons(rows,cases),all_missing_outputs_retained=True,
        frozen_sha256=sha256(out/'frozen_readouts.json'),manifest_sha256=protocol['manifest_sha256'],
        prediction_count=len(rows),predictions_sha256=sha256(dest/'predictions.jsonl'),
        warning='Static positive pointing; common frozen v5 fine decoder for all primary v7 scores. Exploratory paired bootstrap, no multiplicity correction. All checks/choices use training and validation only.')
    write_json(dest/'summary.json',summary)
    print('SUMMARY '+split+' '+json.dumps(dict(primary=summary['primary'],hits=methods[summary['primary']]['hits'],n=len(cases))),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=Path('runs/semantic_head_v7'))
    p.add_argument('--device',default='cuda:0');p.add_argument('--split',choices=['diagnostic','test','both'],default='both')
    args=p.parse_args();torch.set_num_threads(4)
    for split in (['diagnostic','test'] if args.split=='both' else [args.split]): evaluate_split(args.output,split,args.device)


if __name__=='__main__':main()
