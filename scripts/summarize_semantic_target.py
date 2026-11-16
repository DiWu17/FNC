"""Unified paired comparison, with local test scores separate from literature."""
import argparse
import json
from pathlib import Path

import numpy as np

from clickagent.action_benchmark import write_json
from clickagent.public_evaluation import quantization_oracle
from run_target_token_experiment import read_json,read_jsonl,key


def wilson(hits,n):
    if not n:return [None,None]
    z=1.96;p=hits/n;den=1+z*z/n
    center=(p+z*z/(2*n))/den
    radius=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return [float(center-radius),float(center+radius)]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('runs/semantic_target_v2'))
    args=parser.parse_args();output=args.output
    cases=[c for c in read_jsonl(output/'manifest.jsonl') if c['experiment_split']=='test']
    data=read_json(output/'summary.json')
    rows=read_jsonl(output/'predictions.jsonl')
    adapted=output/'adapted_readouts/predictions.jsonl'
    if adapted.exists():
        rows.extend(read_jsonl(adapted))
    ui=output/'ui_tars/summary.json'
    if ui.exists():
        s=read_json(ui)
        if not s['complete']:raise ValueError('UI-TARS baseline incomplete')
        uirows=read_jsonl(output/'ui_tars/predictions.jsonl')
        assert {r['sample_id'] for r in uirows}=={c['sample_id'] for c in cases}
        rows.extend(dict(r,method='ui_tars_1_5_7b') for r in uirows)
    lookup={method:{r['sample_id']:r for r in rows if r['method']==method} for method in {r['method'] for r in rows}}
    methods={}
    for name,group in lookup.items():
        assert set(group)=={c['sample_id'] for c in cases}
        hits=sum(r['hit'] for r in group.values())
        methods[name]={'n':len(group),'hits':hits,'accuracy':hits/len(group),'wilson95_descriptive':wilson(hits,len(group)),
            'datasets':{ds:{'n':sum(c['dataset']==ds for c in cases),'hits':sum(group[c['sample_id']]['hit'] for c in cases if c['dataset']==ds)} for ds in sorted({c['dataset'] for c in cases})}}
    oracle=[];availability={'original':0,'compact':0};examples=[]
    for c in cases:
        for v in availability:
            with np.load(output/'features'/v/f"{key(c['sample_id'])}.npz",allow_pickle=False) as z:
                meta=json.loads(str(z['metadata']))
            availability[v]+=meta['target_available']
            if v==data['selected_on_dev']['variant']:
                oracle.append(quantization_oracle(c['target_geometry'],meta['grid_shape'],c['image_size']))
        before=lookup['v1_original_target_first'][c['sample_id']]['hit']
        after=lookup['optimized'][c['sample_id']]['hit']
        if before!=after:examples.append({'sample_id':c['sample_id'],'kind':'gain' if after else 'loss','instruction':c['instruction']})
    rng=np.random.default_rng(20261031)
    clusters=[np.asarray([i for i,c in enumerate(cases) if c['dataset']==ds]) for ds in sorted({c['dataset'] for c in cases})]
    target=np.asarray([lookup['optimized'][c['sample_id']]['hit'] for c in cases],int)
    comparisons={}
    for reference in ('selected_prompt_prefill','v1_original_target_mean','v1_original_target_first','qwen_native_completed_lax','ui_tars_1_5_7b'):
        if reference not in lookup:continue
        baseline=np.asarray([lookup[reference][c['sample_id']]['hit'] for c in cases],int)
        delta=target-baseline
        boot=[]
        for _ in range(5000):
            idx=np.concatenate([rng.choice(group,len(group),replace=True) for group in clusters])
            boot.append(delta[idx].mean())
        comparisons[reference]={'gains':int((delta>0).sum()),'losses':int((delta<0).sum()),
            'accuracy_difference':float(delta.mean()),'paired_stratified_bootstrap95':np.percentile(boot,[2.5,97.5]).tolist()}
    adapted_comparisons={}
    if 'adapted_target' in lookup:
        adapted_target=np.asarray([lookup['adapted_target'][c['sample_id']]['hit'] for c in cases],int)
        for reference in ('optimized','adapted_prefill','qwen_native_completed_lax','ui_tars_1_5_7b'):
            if reference not in lookup:continue
            baseline=np.asarray([lookup[reference][c['sample_id']]['hit'] for c in cases],int)
            delta=adapted_target-baseline
            boot=[]
            for _ in range(5000):
                idx=np.concatenate([rng.choice(group,len(group),replace=True) for group in clusters])
                boot.append(delta[idx].mean())
            adapted_comparisons[reference]={'gains':int((delta>0).sum()),'losses':int((delta<0).sum()),
                'accuracy_difference':float(delta.mean()),'paired_stratified_bootstrap95':np.percentile(boot,[2.5,97.5]).tolist()}
    report={'selected_on_development':data['selected_on_dev'],'test_n':len(cases),'methods':methods,
        'paired_comparisons':comparisons,'adapted_readout_paired_comparisons':adapted_comparisons,
        'adapted_readout_training':read_json(output/'adapted_readouts/frozen_weights.json') if adapted.exists() else None,
        'available_target_names':availability,
        'grid_center_quantization_ceiling':{'hits':int(sum(oracle)),'n':len(oracle),
            'datasets':{ds:sum(value for c,value in zip(cases,oracle) if c['dataset']==ds) for ds in sorted({c['dataset'] for c in cases})}},
        'different_from_original_first_examples':examples,
        'valid_coordinate_outputs':{m:sum(r['point'] is not None for r in lookup[m].values()) for m in ('qwen_native_strict','qwen_native_completed_lax','ui_tars_1_5_7b') if m in lookup},
        'note':'Same screenshot cohort and pixel budget; prompts/architectures differ. Confidence intervals are descriptive, not app-cluster guarantees. Official benchmark scores are separate.'}
    dest=output/'comparison';dest.mkdir(exist_ok=True)
    write_json(dest/'summary.json',report)
    order=['selected_prompt_prefill','v1_original_target_mean','optimized',
        'optimized_layer15_mean','adapted_prefill','adapted_target',
        'qwen_native_strict','qwen_native_completed_lax','ui_tars_1_5_7b']
    labels={'selected_prompt_prefill':'Prefill / old head','v1_original_target_mean':'Target phrase / old head',
        'optimized':'Target first token / old head','optimized_layer15_mean':'Target first token / layer 15 mean',
        'adapted_prefill':'Prefill / dev-fitted head','adapted_target':'Target first token / dev-fitted head',
        'qwen_native_strict':'Qwen3-VL native / strict','qwen_native_completed_lax':'Qwen3-VL native / complete lax',
        'ui_tars_1_5_7b':'UI-TARS-1.5-7B native'}
    lines=['# Semantic target readout: matched test comparison','',f"Selected on dev: {data['selected_on_dev']}",
        '',f"Fresh image-disjoint test: {len(cases)}. Spatial center ceiling: {sum(oracle)}/{len(oracle)}.",
        '', '| Method | Correct / total | Accuracy |','|---|---:|---:|']
    for method in order:
        if method in methods:
            m=methods[method];lines.append(f"| {labels[method]} | {m['hits']}/{m['n']} | {m['accuracy']:.1%} |")
    (dest/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names=[n for n in order if n in methods]
    fig,ax=plt.subplots(figsize=(11,5))
    y=np.arange(len(names));x=[100*methods[n]['accuracy'] for n in names]
    ax.barh(y,x,color=['#208c70' if n in ('optimized','adapted_target') else '#54769b' for n in names])
    ax.set_yticks(y,[labels[n] for n in names]);ax.invert_yaxis();ax.set_xlim(0,100)
    ax.axvline(100*sum(oracle)/len(oracle),color='#96722b',linestyle='--',label='Grid-center ceiling (attention methods only)')
    ax.legend(loc='lower right',fontsize=8)
    for i,n in enumerate(names):ax.text(x[i]+.8,i,f"{methods[n]['hits']}/{methods[n]['n']} ({x[i]:.1f}%)",va='center')
    ax.set_xlabel('Point-in-target accuracy (%)');ax.set_title('Fresh matched screenshot test, max_pixels=262144')
    fig.tight_layout();fig.savefig(dest/'comparison.png',dpi=160);fig.savefig(dest/'comparison.pdf');plt.close(fig)
    print(json.dumps(report,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
