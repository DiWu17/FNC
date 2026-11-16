"""Posthoc fixed-weight permutation audit, registered before its own outputs."""
import os
os.environ.setdefault('PYTHONDONTWRITEBYTECODE','1')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from datetime import datetime,timezone
from pathlib import Path
import gc
import json
import sys
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import numpy as np
import torch
import run_bottleneck_v19 as probe
from clickagent.bottleneck_v19 import shuffle_indices

OUT=probe.OUT/'analysis/permutation_audit'


def status(phase,**fields):
    row=dict(phase=phase,complete=phase=='complete',updated_at_utc=probe.now(),**fields)
    probe.write(OUT/'status.json',row);print(json.dumps(row,ensure_ascii=False),flush=True)


def register():
    path=OUT/'protocol.json'
    if path.exists():
        spec=probe.read(path)
        for p,h in spec['hashes'].items():assert probe.sha(p)==h,p
        return
    assert not list((OUT/'evaluation').glob('*.json'))
    paths=[Path(__file__),probe.OUT/'completion.json',probe.OUT/'protocol.json',
           probe.OUT/'models/frozen.json',probe.OUT/'features/index.json',probe.OUT/'normalization.pt',
           probe.OUT/'results.json',ROOT/'src/clickagent/bottleneck_v19.py']
    probe.write(path,dict(registered_at_utc=probe.now(),hashes={str(p):probe.sha(p) for p in paths},
        scope='Posthoc diagnostic motivated by completed V19 train/development gap. Not part of its four primary comparisons; no new confirmation claim.',
        models='Existing three visual fits and three visual_shuffled fits, immutable checkpoints; no training or model selection.',
        interventions=dict(visual_to_fixed_shuffle='Apply the pre-existing D control permutation only at inference to the aligned C fits.',
                           shuffled_to_fresh_shuffle='Replace D training-time fixed permutation at inference with three preregistered alternate permutations; suffix |posthoc_reshuffle_0/1/2 passed to the unchanged shuffle_indices.'),
        isolation='Each intervention preserves within-query/view feature multisets and geometric masks. No candidate change, new features, VLM call or fresh280 access.',
        evaluation='All train2973, old172 and new300 development. Replay original C/D hit vectors exactly before reporting counterfactuals.',
        uncertainty='Descriptive95% source-stratified RGB-cluster bootstrap2000 of fixed3-fit means. D alternate input variants averaged over3 fixed permutations; not9 trained models, an ensemble or extra independent samples.',
        interpretation='Sensitivity on training plus little development benefit is consistent with fitting sample-specific associations. It does not identify data noise, distribution shift or all possible representation causes separately.'))


def intervals(diff,entries):
    groups={}
    for i,e in enumerate(entries):groups.setdefault(e['source'],{}).setdefault(e['group'],[]).append(i)
    strata=[list(v.values()) for v in groups.values()];rng=np.random.default_rng(20261903)
    draws=[[i for clusters in strata for j in rng.integers(0,len(clusters),len(clusters)) for i in clusters[j]] for _ in range(2000)]
    return dict(delta_pp=100*float(diff.mean()),ci95_pp=(100*np.quantile([diff[ix].mean() for ix in draws],[.025,.975])).tolist(),descriptive_posthoc=True)


@torch.inference_mode()
def main():
    register();probe.verify();torch.set_num_threads(2);device='cuda:0';torch.cuda.set_device(device)
    receipt=probe.read(probe.OUT/'completion.json')
    for p,h in receipt['artifact_sha256'].items():assert probe.sha(p)==h,p
    frozen=probe.read(probe.OUT/'models/frozen.json');models={}
    for arm in ['visual','visual_shuffled']:
        for s in [1,2,3]:
            name=f'{arm}_s{s}';row=frozen['models'][name]
            assert probe.sha(row['path'])==row['sha256']
            saved=torch.load(row['path'],weights_only=True,map_location='cpu')
            model=probe.model_for(arm).to(device).eval();model.load_state_dict(saved['state_dict']);models[name]=model
    norm=torch.load(probe.OUT/'normalization.pt',weights_only=True);indices=probe.read(probe.OUT/'data/indices.json');results={}
    for cohort in ['train','old_val','new_development']:
        path=OUT/f'evaluation/{cohort}.json'
        if path.exists():results[cohort]=probe.read(path);continue
        status('loading',cohort=cohort);records=probe.load_records(cohort);cache=probe.BatchCache(records,norm,device)
        permutations=[[torch.tensor(shuffle_indices(r['roi_mask'],r['sample_id']+f'|posthoc_reshuffle_{draw}'),device=cache.storage) for r in records] for draw in range(3)]
        values={k:[[],[],[]] for k in ['visual','visual_to_shuffle','shuffled','reshuffle_0','reshuffle_1','reshuffle_2']}
        for start in range(0,len(records),8):
            ix=list(range(start,min(start+8,len(records))));args,lengths,labels=cache.batch(ix);fixed=cache.batch(ix,True)[0]
            alternate=[]
            for draw in range(3):
                local=torch.cat([cache.data[i]['roi'][permutations[draw][i],torch.arange(4,device=cache.storage)[None,:]] for i in ix]).to(device)
                alternate.append(dict(args,roi=local))
            for s in [1,2,3]:
                for name,inputs in [('visual',args),('visual_to_shuffle',fixed),('shuffled',fixed)]+[(f'reshuffle_{d}',alternate[d]) for d in range(3)]:
                    arm='visual' if name.startswith('visual') else 'visual_shuffled'
                    chunks=probe.forward(models[f'{arm}_s{s}'],inputs).split(lengths)
                    values[name][s-1].extend(bool(len(v) and y[int(v.argmax())]) for v,y in zip(chunks,labels))
            if start%80==0 or start+8>=len(records):status('evaluating',cohort=cohort,completed=min(start+8,len(records)),total=len(records))
        original=probe.read(probe.OUT/f'evaluation/{cohort}.json')
        for s in [1,2,3]:
            assert values['visual'][s-1]==original['hit_vectors'][f'visual_s{s}']
            assert values['shuffled'][s-1]==original['hit_vectors'][f'visual_shuffled_s{s}']
        arrays={k:np.asarray(v,float) for k,v in values.items()}
        alternate_mean=np.mean([arrays[f'reshuffle_{d}'] for d in range(3)],axis=0)
        methods={k:dict(accuracy_percent=100*float(v.mean()),seed_hits=v.sum(1).tolist()) for k,v in arrays.items()}
        methods['reshuffle_mean']=dict(accuracy_percent=100*float(alternate_mean.mean()),fixed_fits=3,input_permutations=3,ensemble=False)
        result=dict(complete=True,n=len(records),methods=methods,hit_vectors=values,
                    comparisons=dict(C_shuffle_minus_aligned=intervals((arrays['visual_to_shuffle']-arrays['visual']).mean(0),indices[cohort]),
                                     D_reshuffle_minus_fixed=intervals((alternate_mean-arrays['shuffled']).mean(0),indices[cohort])),
                    original_C_D_point_hit_replay=True,scope='Posthoc fixed-weight diagnosis only')
        probe.write(path,result);results[cohort]=result
        del cache,records,args,fixed,alternate,permutations;gc.collect();torch.cuda.empty_cache()
    lines=['V19 事后固定权重置换核查：没有新训练，也没有使用fresh280。',
           'C=正确对应局部特征训练，D=固定打乱对应训练；原C/D六个模型保持不变。',
           '重新打乱D取预先固定的三种输入置换均值，不是九个训练种子或集成。']
    for cohort,r in results.items():
        lines += ['',cohort]
        for name in ['visual','visual_to_shuffle','shuffled','reshuffle_mean']:lines.append(f'{name}: {r["methods"][name]["accuracy_percent"]:.2f}%')
        for name,c in r['comparisons'].items():lines.append(f'{name}: {c["delta_pp"]:+.2f}pp, 描述性95%区间{c["ci95_pp"]}')
    probe.write(OUT/'results.json',dict(complete=True,cohorts=results,scope='Posthoc descriptive; original registered V19 results untouched'))
    (OUT/'readout.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    register();probe.verify()
    paths=[OUT/'protocol.json',OUT/'results.json',OUT/'readout.txt']+[OUT/f'evaluation/{c}.json' for c in results]
    probe.write(OUT/'completion.json',dict(complete=True,completed_at_utc=probe.now(),artifact_sha256={str(p):probe.sha(p) for p in paths},no_training=True,mainline_unchanged=True))
    status('complete');print('\n'.join(lines),flush=True)


if __name__=='__main__':
    try:main()
    except BaseException:status('failed',error=traceback.format_exc());raise
