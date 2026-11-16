"""Supplementary exact-text diagnostic and integrity checks; no fitting/selection."""
from pathlib import Path
import json
import sys
import math
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
import numpy as np
import torch
import run_transfer_alignment_v20 as run


def main():
    out=run.OUT;run.verify();completed=run.read(out/'completion.json');assert completed['complete']
    for p,h in completed['artifact_sha256'].items():assert run.sha(p)==h
    frozen=run.read(out/'models/frozen.json');results=run.read(out/'results.json');plan=run.read(out/'instruction_strata_plan.json')
    assert plan['before_any_new_head_development_outputs'] and plan['primary_unchanged']
    fit_diagnostics={};verified=0
    for name,row in frozen['models'].items():
        assert run.sha(row['path'])==row['sha256'] and row['steps']==2976
        assert row['fitting_queries']==2878 and not row['development_used_for_selection']
        state=torch.load(row['path'],weights_only=True,map_location='cpu')['state_dict']
        anchor=float(((state['region.weight']-state['region_anchor']).square().sum()+(state['query.weight']-state['query_anchor']).square().sum())/32)
        fit_diagnostics[name]=dict(residual_scale=4*float(state['alpha'].tanh()),anchor_distance=anchor,
                                  elapsed_seconds=row['history'][-1]['elapsed_seconds'],seed=row['seed'])
        verified+=1
    for row in run.read(run.prior.MAIN/'models/frozen.json')['models'].values():assert run.sha(row['path'])==row['sha256']
    replay=0;strata={}
    for cohort in run.COHORTS:
        e=run.read(out/f'evaluation/{cohort}.json');old=run.read(run.prior.OUT/f'evaluation/{cohort}.json')
        for s in [1,2,3]:
            name=f'original_s{s}';assert e['hit_vectors'][name]==old['hit_vectors'][name];replay+=e['n']
        for arm in ['original']+run.ARMS:
            value=np.array([e['hit_vectors'][f'{arm}_s{s}'] for s in [1,2,3]],float)
            assert value.shape==(3,e['n'])
            assert abs(100*value.mean()-results['cohorts'][cohort]['methods'][arm]['accuracy_percent'])<1e-10
        if cohort=='train':continue
        strata[cohort]={}
        for label,key in [('seen','seen_exact_normalized_instruction_indices'),('unseen','unseen_exact_normalized_instruction_indices')]:
            ix=plan['cohorts'][cohort][key]
            methods={}
            for arm in ['original']+run.ARMS:
                v=np.array([e['hit_vectors'][f'{arm}_s{s}'] for s in [1,2,3]],float)[:,ix]
                methods[arm]=dict(accuracy_percent=100*float(v.mean()),seed_hits=v.sum(1).astype(int).tolist())
            strata[cohort][label]=dict(n=len(ix),methods=methods,combined_minus_original_pp=methods['paired_regularized']['accuracy_percent']-methods['original']['accuracy_percent'])
    assert replay==10335
    report=dict(complete=True,analyzed_at_utc=run.now(),plan_sha256=run.sha(out/'instruction_strata_plan.json'),
                descriptive_exact_text_strata=strata,fit_diagnostics=fit_diagnostics,
                limitation='Literal string novelty is not semantic novelty or dataset holdout; all cohorts are reused development.',
                parent_results_sha256=run.sha(out/'results.json'))
    run.write(out/'supplementary.json',report)
    lines=['V20补充：预先登记的完整指令文本重复/未重复分层（仅描述性）。']
    for cohort,groups in strata.items():
        for label,r in groups.items():
            lines.append(f'{cohort}/{label}: n={r["n"]}, 原版{r["methods"]["original"]["accuracy_percent"]:.2f}%, 组合{r["methods"]["paired_regularized"]["accuracy_percent"]:.2f}%, 差{r["combined_minus_original_pp"]:+.2f}pp')
    lines.append('不能把文本未重复解释为新的语义任务、指令家族或未见来源。')
    lines.append(f'12新检查点验证通过；原版命中向量重放{replay}/{replay}一致。')
    (out/'supplementary.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    verification=dict(complete=True,verified_at_utc=run.now(),new_checkpoint_hashes_verified=verified,
        original_hit_vector_replay=replay,primary_metric_recomputation=True,original_checkpoints_unchanged=True,
        inputs={str(out/p):run.sha(out/p) for p in ['protocol.json','completion.json','models/frozen.json','results.json','instruction_strata_plan.json']},
        supplementary_sha256=run.sha(out/'supplementary.json'))
    run.write(out/'verification.json',verification)
    print('\n'.join(lines),flush=True)


if __name__=='__main__':main()
