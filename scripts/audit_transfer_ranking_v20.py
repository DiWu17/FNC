"""Posthoc descriptive check: average compatibility versus hard distractors."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
import numpy as np
import torch
import run_transfer_alignment_v20 as run
from clickagent.transfer_alignment_v20 import TransferMatcher


@torch.inference_mode()
def main():
    run.verify();out=run.OUT;assert run.read(out/'completion.json')['complete']
    plan=out/'ranking_audit_plan.json'
    if not plan.exists():
        run.write(plan,dict(registered_at_utc=run.now(),posthoc_after_primary_results=True,no_new_fit_or_selection=True,
            parent_results_sha256=run.sha(out/'results.json'),
            diagnostic='For matching/paired global+local-view cosine scores: positive-vs-mean-negative and best-positive-vs-best-negative margins, branch-only top1, signed residual top1, and original errors repaired/broken. Descriptive only, no extra significance tests or claims of unique bottleneck.'))
    device='cuda:0';torch.cuda.set_device(0);torch.set_num_threads(2)
    frozen=run.read(out/'models/frozen.json');models={}
    for arm in ['matching','paired']:
        for s in [1,2,3]:
            name=f'{arm}_s{s}';row=frozen['models'][name];assert run.sha(row['path'])==row['sha256']
            model=TransferMatcher().to(device).eval();model.load_state_dict(torch.load(row['path'],weights_only=True,map_location='cpu')['state_dict']);models[name]=model
    result={}
    for cohort in run.COHORTS:
        saved=run.load_compact(cohort,device);rows=saved['rows'];ev=run.read(out/f'evaluation/{cohort}.json');stats={}
        for name,model in models.items():
            s=int(name[-1])-1;mm=[];hm=[];branch=[];signed=[];repaired=broken=0;conditional=0
            for start in range(0,len(rows),16):
                args,lengths,selected,baseline=run.batch(rows,list(range(start,min(start+16,len(rows)))),s)
                residual,rr,qq=model(**args)
                cosine=((rr*qq).sum(-1)*args['roi_mask']).sum(1)/args['roi_mask'].sum(1).clamp_min(1)
                for j,(r,c,z,b) in enumerate(zip(selected,cosine.split(lengths),residual.split(lengths),baseline.split(lengths))):
                    i=start+j;y=r['labels'];old=ev['hit_vectors'][f'original_s{s+1}'][i];new=ev['hit_vectors'][name][i]
                    repaired+=int(new and not old);broken+=int(old and not new)
                    if r['n'] and bool(y.any()):
                        conditional+=1;branch.append(bool(y[int(c.argmax())]));signed.append(bool(y[int(z.argmax())]))
                        if bool((~y).any()):
                            mm.append(float(c[y].mean()-c[~y].mean()));hm.append(float(c[y].max()-c[~y].max()))
            stats[name]=dict(n=ev['n'],reachable=conditional,cosine_mean_positive_margin=float(np.mean(mm)),
                cosine_best_positive_minus_best_negative=float(np.mean(hm)),cosine_branch_conditional_top1_percent=100*float(np.mean(branch)),
                signed_residual_conditional_top1_percent=100*float(np.mean(signed)),
                original_wrong_repaired=repaired,original_right_broken=broken,
                residual_scale=4*float(model.alpha.tanh()))
        result[cohort]=stats
    summary={}
    for cohort,stats in result.items():
        summary[cohort]={}
        for arm in ['matching','paired']:
            rows=[stats[f'{arm}_s{s}'] for s in [1,2,3]]
            summary[cohort][arm]={k:float(np.mean([r[k] for r in rows])) for k in ['cosine_mean_positive_margin','cosine_best_positive_minus_best_negative','cosine_branch_conditional_top1_percent','signed_residual_conditional_top1_percent','original_wrong_repaired','original_right_broken']}
    run.write(out/'ranking_audit.json',dict(complete=True,posthoc=True,descriptive_only=True,plan_sha256=run.sha(plan),
        cohorts=result,three_fit_means=summary,limitations='Cached native target summary and regions; branch-only scores are not a separately trained standalone ranker. Signed residual can invert compatibility in some seeds. No new independent evidence.'))
    print(summary,flush=True)


if __name__=='__main__':main()
