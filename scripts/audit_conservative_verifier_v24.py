"""Post-run replay and ceiling diagnosis; does not change registered V24 policy."""
import os
os.environ.setdefault('PYTHONDONTWRITEBYTECODE', '1')
import hashlib
import json
from pathlib import Path
import sys
from datetime import datetime, timezone
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import numpy as np
import run_conservative_verifier_v24 as run
from clickagent.conservative_verifier_v24 import (MARGIN,combine,features,fit_gate,predict_utility)
from run_visual_feedback_v23 import read,write,sha,hit


def main():
    out=run.OUT;run.verify()
    completion=read(out/'completion.json');assert completion['complete']
    for path,h in completion['artifact_sha256'].items():assert sha(path)==h,path
    cs=read(out/'data/public.json');labels=read(out/'data/labels.json')
    ps=read(out/'data/proposals.json');pred=read(out/'predictions.json');rows=pred['records']
    assert len(rows)==600 and len({(r['sample_id'],r['model']) for r in rows})==600
    prior={c['sample_id']:c for c in read(run.V21/'data/public.json')}
    case={c['sample_id']:c for c in cs}
    gates=read(out/'gate_models.json')['fold_models'];tasks=read(out/'data/tasks.json')
    calls={t['task_id']:read(out/f'calls/{t["task_id"]}.json') for t in tasks}
    protocol=sha(out/'protocol.json');geometry_checks=0;decision_replays=0
    for task in tasks:
        saved=calls[task['task_id']];c=case[task['sample_id']]
        assert saved['task']==task and saved['protocol_sha256']==protocol
        assert saved['image_sha256']==c['image_sha256']
        g=saved['output']['geometry'];assert g['pool']==task['pool']
        assert not g['ground_truth_in_model_inputs']
        for j,crop in enumerate(g['crops']):
            assert crop['point']==task['pool'][saved['output']['order'][j]]
            x,y,x1,y1=crop['box'];px,py=crop['point'];w,h=c['image_size']
            assert 0<=x<=px<x1<=w and 0<=y<=py<y1<=h
            assert crop['render']['source_size']==[x1-x,y1-y]
            expected=[px-x,py-y]
            # Native256-source-pixel crops; tiny screenshots are letterboxed.
            meta=crop['render'];rw,rh=meta['resized_size'];ox,oy=meta['offset']
            center=[ox+expected[0]*rw/(x1-x),oy+expected[1]*rh/(y1-y)]
            assert np.allclose(meta['marked_centers'][0],center)
            geometry_checks+=1
    old={(r['sample_id'],r['model'],r['arm']):r for r in read(run.PRIOR/'predictions.json')['rows']}
    for r in rows:
        sid=r['sample_id'];c=case[sid];model=ps[sid]['models'][r['model']];pool=model['pool']
        assert prior[sid]==c,'MVP and new visual queries differ'
        assert r['points']['baseline']==old[sid,r['model'],'baseline']['point']
        assert r['points']['v23_verify']==old[sid,r['model'],'marked_verify']['point']
        if pool:
            canonical=sorted(pool);obs=[]
            for rev in [0,1]:
                o=calls[model['signature']+'_'+str(rev)]['output']
                obs.append(dict(scores=o['scores'],order=[pool.index(canonical[i]) for i in o['order']]))
            combined=combine(pool,obs)
            assert combined==r['combined']
            assert combined['unconditional_point']==r['points']['expanded_ungated']
            assert combined['fixed_point']==r['points']['expanded_fixed']
            manual=combined['proposed_change'] and len(set(combined['per_order_choice']))==1
            manual=manual and combined['min_winner_runnerup_margin']>=MARGIN
            assert manual==combined['fixed_accept']
            f=features(combined,pool,ps[sid]['points'],model['scores'],c['image_size'])
            assert np.allclose(f,r['features'],atol=1e-12,rtol=0)
        base=hit(r['points']['baseline'],c,labels[sid]);un=hit(r['points']['expanded_ungated'],c,labels[sid])
        assert r['utility']==int(un)-int(base)
        decision_replays+=len(r['points'])
    for saved in gates:
        fold=saved['fold'];training=[r for r in rows if r['fold']!=fold];testing=[r for r in rows if r['fold']==fold]
        assert not {r['sample_id'] for r in training}&{r['sample_id'] for r in testing}
        assert not {tuple(r['group']) for r in training}&{tuple(r['group']) for r in testing}
        fitted=fit_gate(training)
        assert np.allclose(fitted['coefficients'],saved['model']['coefficients'],atol=1e-12,rtol=0)
        for r in testing:
            value=predict_utility(fitted,r['features']);assert abs(value-r['predicted_utility_oof'])<1e-12
            accept=bool(r['proposed_change'] and r['combined']['order_agrees'] and value>0)
            assert accept==r['learned_accept_oof']
            point=r['points']['expanded_ungated'] if accept else r['points']['baseline']
            assert point==r['points']['expanded_learned_oof']
    summaries={}
    for cohort in ['new_development','old_val','all']:
        selected=[c for c in cs if cohort=='all' or c['cohort']==cohort];ids={c['sample_id'] for c in selected}
        report={}
        for family in run.FAMILIES:
            rr=[r for r in rows if r['sample_id'] in ids and r['model'].startswith(family+'_')]
            base=np.array([hit(r['points']['baseline'],case[r['sample_id']],labels[r['sample_id']]) for r in rr])
            raw=np.array([hit(r['points']['expanded_ungated'],case[r['sample_id']],labels[r['sample_id']]) for r in rr])
            mv=np.array([hit(pred['mvp'][r['sample_id']],case[r['sample_id']],labels[r['sample_id']]) for r in rr])
            report[family]=dict(
                perfect_keep_vs_ungated_gate_oracle_percent=100*float((base|raw).mean()),
                full_pool_oracle_percent=100*float(np.mean([r['full_oracle'] for r in rr])),
                nine_pool_oracle_percent=100*float(np.mean([r['new9_oracle'] for r in rr])),
                mvp_percent=100*float(mv.mean()),
                mean_potential_new9_repairs_on_mvp_errors=sum(r['new9_oracle'] and not m for r,m in zip(rr,mv))/3,
                mean_mvp_correct_but_new9_unreachable=sum(m and not r['new9_oracle'] for r,m in zip(rr,mv))/3,
                mean_new9_repairs_unrealized=sum(r['new9_oracle'] and not u for r,u in zip(rr,raw))/3,
                order_disagreement_count=sum(bool(r['proposed_change']) and not r['combined']['order_agrees'] for r in rr),
                policy={})
            for arm in ['expanded_ungated','expanded_fixed','expanded_learned_oof']:
                v=np.array([hit(r['points'][arm],case[r['sample_id']],labels[r['sample_id']]) for r in rr])
                report[family]['policy'][arm]=dict(mean_repairs=float((v&~base).sum()/3),mean_harms=float((~v&base).sum()/3),
                    mean_ungated_repairs_retained=float((v&raw&~base).sum()/3),mean_ungated_harms_prevented=float((v&~raw&base).sum()/3),
                    mean_point_changes=sum(r['points'][arm]!=r['points']['baseline'] for r in rr)/3)
        summaries[cohort]=report
    write(out/'ceiling_and_gate_diagnosis.json',dict(posthoc=True,not_deployable_oracle=True,cohorts=summaries))
    reg=read(ROOT/'CURRENT_RESEARCH.json');before=read(out/'registry_before.json')
    assert reg['run_id']==before['run_id'] and reg['models']==before['models']
    audit=dict(complete=True,completed_at_utc=datetime.now(timezone.utc).isoformat(),protocol_sha256=protocol,
        source_sha256=sha(Path(__file__)),decision_replays=decision_replays,geometry_checks=geometry_checks,
        calls_verified=len(tasks),gate_folds_refitted=5,heldout_rgb_or_application_overlap=0,
        original_baseline_parity=600,mvp_public_query_parity=100,mainline_unchanged=True,
        primary_artifact_hashes_verified=True)
    write(out/'final_audit.json',audit)
    print(json.dumps(audit,ensure_ascii=False))
    print(json.dumps(summaries['new_development'],ensure_ascii=False,indent=2))


if __name__=='__main__':main()
