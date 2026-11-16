"""Post-score diagnostics only; these oracle labels never choose an action."""
import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from clickagent.action_benchmark import write_json
from clickagent.action_grounding import _pixel_centers
from clickagent.public_evaluation import point_in_geometry,quantization_oracle
from clickagent.semantic_target import temporal_map
from run_semantic_target import load_feature
from run_target_token_experiment import read_json,read_jsonl,overlap_cells


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=Path('runs/semantic_target_v2'))
    output=p.parse_args().output
    assert read_json(output/'completion.json')['complete']
    choice=read_json(output/'frozen_selection.json')['selected']
    cases=[c for c in read_jsonl(output/'manifest.jsonl') if c['experiment_split']=='test']
    predictions=read_jsonl(output/'predictions.jsonl')
    adapted=output/'adapted_readouts/predictions.jsonl'
    if adapted.exists():predictions.extend(read_jsonl(adapted))
    methods=('selected_prompt_prefill','optimized','adapted_prefill','adapted_target')
    lookup={(r['method'],r['sample_id']):r for r in predictions}
    buckets={m:Counter() for m in methods if (m,cases[0]['sample_id']) in lookup}
    temporal=[];records=[]
    for c in cases:
        raw,meta=load_feature(output,c,choice['variant'])
        target=temporal_map(raw,meta,choice['policy'])
        grid=meta['grid_shape'];w,h=c['image_size']
        ceiling=quantization_oracle(c['target_geometry'],grid,c['image_size'])
        mask=overlap_cells(c['target_geometry'],grid)
        x,y=_pixel_centers(grid,c['image_size'])
        center_mask=np.asarray([point_in_geometry([a/w,b/h],c['target_geometry']) for a,b in zip(x,y)])
        record={'sample_id':c['sample_id'],'dataset':c['dataset'],'instruction':c['instruction'],
            'target_text':meta['target_text'],'target_available':meta['target_available'],
            'grid_center_can_hit':ceiling,'grid_shape':grid,'image_size':c['image_size']}
        for method,bucket in buckets.items():
            row=lookup[method,c['sample_id']]
            if row['hit']:kind='correct'
            elif row['point'] is None:kind='no_target_name'
            elif not ceiling:kind='no_representable_grid_center'
            elif row['cell_overlap']:kind='selected_cell_overlaps_but_center_misses'
            else:kind='selected_wrong_cell'
            bucket[kind]+=1;record[method]=dict(row,failure_stage=kind)
        if target is not None:
            metrics={}
            for stage,attention in (('prefill',raw[0,:32]),('target',target[:32])):
                heat=attention.mean(0);mass=float(heat.sum())
                metrics[stage]={'image_mass':mass,'target_overlap_relative_mass':float(heat[mask].sum()/mass) if mass else 0.}
            temporal.append(metrics)
            record['any_individual_head_peak_hits']=bool(center_mask[np.argmax(target,axis=1)].any())
        records.append(record)
    summary={'n_test':len(cases),'buckets':{m:dict(b) for m,b in buckets.items()},
        'paired_attention_diagnostic_n':len(temporal),
        'raw_layer15_attention':{stage:{metric:float(np.mean([r[stage][metric] for r in temporal])) for metric in ('image_mass','target_overlap_relative_mass')} for stage in ('prefill','target')},
        'target_relative_mass_increased':sum(r['target']['target_overlap_relative_mass']>r['prefill']['target_overlap_relative_mass'] for r in temporal),
        'any_individual_head_peak_hits':sum(r.get('any_individual_head_peak_hits',False) for r in records),
        'note':'Labels and individual-head oracle used only for post-score diagnosis, never for inference or tuning. Cell overlap is not point accuracy; absent names remain failures.'}
    dest=output/'diagnostics';dest.mkdir(exist_ok=True)
    write_json(dest/'summary.json',summary)
    (dest/'cases.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records),encoding='utf-8')
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
