"""Additional retrospective checks: repair supervision and target consistency."""
import os
os.environ.setdefault('PYTHONDONTWRITEBYTECODE','1')
from pathlib import Path
from datetime import datetime,timezone
from collections import Counter
import sys,json
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import numpy as np
import run_candidate_identity_v27 as run
from clickagent.candidate_identity_v27 import retain
from run_location_selection_v26 import geometry_bins
from run_visual_feedback_v23 import read,write,sha,hit
OUT=run.OUT/'why_analysis_20261109'


def main():
    write(OUT/'consistency_plan.json',dict(recorded_at_utc=datetime.now(timezone.utc).isoformat(),
        source_sha256=sha(Path(__file__)),scope='Posthoc metadata/label audit; no fitting, no generation, no policy adoption.',
        questions=['How many original-training queries teach novel B-only rescue?',
                   'Do target-bearing framed views sometimes say WAIT?',
                   'Which labels at each view are averaged at candidates?']))
    cases=read(run.V25/'data/public.json');labels=read(run.V25/'data/labels.json')
    cases_out=[];model_out=[];view_rows=[]
    for c in cases:
        sid=c['sample_id'];a=run.load(c);truth=np.asarray([hit(p.tolist(),c,labels[sid]) for p in a['points']],bool);A=a['ai']>=0
        cases_out.append(dict(sample_id=sid,cohort=c['cohort'],source=c['source'],A_empty=not A.any(),
            A_covered=bool(truth[A].any()),U_covered=bool(truth.any()),B_new_positive=bool(truth[~A].any()),
            only_B_can_cover=bool(truth.any() and not truth[A].any())))
        for name in run.NAMES:
            ix=retain(a['U__'+name],a['ai']);aix=np.flatnonzero(A)
            best_A=int(aix[np.asarray(a['A__'+name]).argmax()]) if len(aix) else None
            A_hit=bool(best_A is not None and truth[best_A])
            model_out.append(dict(sample_id=sid,cohort=c['cohort'],model=name,A_hit=A_hit,
                covered_retained=bool(truth[ix].any()),B_new_positive_retained=bool((truth[ix]&(~A[ix])).any()),
                B_only_rescue_retained=bool(not truth[A].any() and truth[ix].any()),
                A_failure_with_B_positive_retained=bool(not A_hit and (truth[ix]&(~A[ix])).any())))
        for vi,v in enumerate(c['views']):
            p=run.V25/f'raw/{run.prior.task_for(c,vi,"context")["task_id"]}.npz'
            with np.load(p,allow_pickle=False) as z:m=json.loads(str(z['metadata']))
            boxes=geometry_bins(c,labels[sid],v['transform'])[2]
            row=dict(sample_id=sid,cohort=c['cohort'],source=c['source'],view=vi,
                has_target_tokens=bool(m['readout_stages']['target']),action=m['action'],
                target=m['ground_fields'].get('target'),container=m['ground_fields'].get('container'),
                location=m['ground_fields'].get('location'),target_visible=bool(boxes),text=m['generated_text'])
            view_rows.append(row)
    summary={}
    for cohort in ['train','new_development','old_val']:
        cc=[r for r in cases_out if r['cohort']==cohort];vv=[r for r in view_rows if r['cohort']==cohort]
        nonempty=[r for r in vv if r['has_target_tokens']]
        s=dict(n_queries=len(cc),A_empty=sum(r['A_empty'] for r in cc),
            A_covered=sum(r['A_covered'] for r in cc),union_covered=sum(r['U_covered'] for r in cc),
            only_B_can_cover=sum(r['only_B_can_cover'] for r in cc),
            any_B_new_positive=sum(r['B_new_positive'] for r in cc),
            B_only_rescue_cases=[r for r in cc if r['only_B_can_cover']],
            views=len(vv),views_with_target=len(nonempty),
            target_bearing_action_counts=dict(Counter(r['action'] for r in nonempty)),
            WAIT_with_visible_target=sum(r['action']=='WAIT' and r['target_visible'] for r in nonempty),per_model={})
        for name in run.NAMES:
            rr=[r for r in model_out if r['cohort']==cohort and r['model']==name]
            s['per_model'][name]={k:sum(r[k] for r in rr) for k in ['A_hit','covered_retained','B_new_positive_retained','B_only_rescue_retained','A_failure_with_B_positive_retained']}
        summary[cohort]=s
    write(OUT/'evidence_consistency.json',dict(summary=summary,case_rows=cases_out,model_rows=model_out,view_rows=view_rows,
        name_difference_does_not_automatically_mean_wrong=True,views_are_correlated_within_RGB=True))
    paths=[Path(__file__),OUT/'consistency_plan.json',OUT/'evidence_consistency.json']
    write(OUT/'consistency_completion.json',dict(complete=True,artifact_sha256={str(p):sha(p) for p in paths},
        source_results_unchanged=True,new_vlm_calls=0))
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
