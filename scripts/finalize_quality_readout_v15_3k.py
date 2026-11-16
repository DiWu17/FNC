"""Repair only the completion audit; registered training/evaluation sources stay sealed.

The registered complete() called .get on details.json, whose schema is a list.
Replay cached predictions against authenticated coordinates/labels, validate both
JSON schemas, and write the missing completion record without model inference.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import numpy as np
import run_quality_readout_v15_3k as run


def validate_details(details, result, model_names):
    run.require(isinstance(details,list),'Details must be a list')
    run.require(isinstance(result,dict) and result.get('complete') is True,'Results must be complete')
    run.require(len(details)==result['n'],'Incomplete details count')
    run.require(len({r['sample_id'] for r in details})==len(details),'Repeated detail IDs')
    run.require(set(result['methods'])==set(model_names),'Incomplete result models')
    for row in details:run.require(set(row['methods'])==set(model_names),'Incomplete detail models')
    fields={'top1':'top1_hits','top4':'top4_coverage','union_top1':'union_top1_hits'}
    for name in model_names:
        for flag,field in fields.items():
            run.require(all(type(r['methods'][name][flag]) is bool for r in details),'Nonboolean hit flag')
            run.require(sum(r['methods'][name][flag] for r in details)==result['methods'][name][field],'Detail/aggregate mismatch')
        run.require(np.isclose(result['methods'][name]['accuracy'],result['methods'][name]['top1_hits']/result['n'],atol=1e-14,rtol=0),'Accuracy mismatch')
    for source,values in result['by_source'].items():
        rows=[r for r in details if r['source']==source]
        run.require(len(rows)==values['n'],'Source denominator mismatch')
        for name in model_names:run.require(sum(r['methods'][name]['top1'] for r in rows)==values['hits'][name],'Source hits mismatch')


def audit_cohort(cohort, frozen):
    out=run.OUT
    result=run.read(out/f'evaluation/{cohort}/results.json')
    details=run.read(out/f'evaluation/{cohort}/details.json')
    validate_details(details,result,frozen['models'])
    entries=run.base.old_entries('val') if cohort=='old_val' else run.read(out/'features/new_reserve/index.json')['cases']
    run.require(result['n']==(172 if cohort=='old_val' else 300),'Wrong validation size')
    run.require({r['sample_id'] for r in details}=={e['sample_id'] for e in entries},'Evaluation membership mismatch')
    lookup={r['sample_id']:r for r in details};vectors={name:[] for name in frozen['models']}
    for e in entries:
        for field,h in [('path','sha256'),('labels_path','labels_sha256')]:
            run.require(run.sha(e[field])==e[h],'Coordinate/label bytes drift')
        with np.load(e['path'],allow_pickle=False) as z:
            points=z['points'].copy();pool=np.sort(z['baseline'])
        labels=np.load(e['labels_path'],allow_pickle=False)
        row=lookup[e['sample_id']]
        run.require(row['source']==e['source'] and row['group']==e['group'] and row['source_group']==e['source_group'],'Grouping drift')
        for name,model in frozen['models'].items():
            p=out/f'predictions/{cohort}/{name}/{run.key(e["sample_id"])}.npz'
            with np.load(p,allow_pickle=False) as z:
                run.require(str(z['checkpoint_sha256'])==model['sha256'] and str(z['base_sha256'])==e['sha256'],'Prediction identity drift')
                scores=z['scores'].copy()
            run.require(scores.shape==(e['n'],) and np.isfinite(scores).all(),'Malformed scores')
            order=pool[np.argsort(-scores[pool],kind='stable')]
            pick=int(order[0]) if len(order) else None;whole=int(scores.argmax()) if len(scores) else None
            value=dict(point=points[pick].tolist() if pick is not None else None,
                top1=bool(pick is not None and labels[pick]),top4=bool(labels[order[:4]].any()),
                union_top1=bool(whole is not None and labels[whole]))
            run.require(value==row['methods'][name],'Prediction/detail replay mismatch')
            vectors[name].append(value['top1'])
    for arm in ('old_matched','old8'):
        for s in (1,2,3):
            a,b=f'expanded_s{s}',f'{arm}_s{s}';expected=result['comparisons'][a+' vs '+b]
            replay=run.paired(vectors[a],vectors[b],entries,bootstrap=True)
            for field in ('n','method_hits','reference_hits','wins','losses','delta','ci95'):
                run.require(np.allclose(replay[field],expected[field],atol=1e-14,rtol=0),'Paired comparison replay mismatch')
            if cohort=='new_reserve':
                grouped=[dict(e,group=e['source_group']) for e in entries]
                ci=run.paired(vectors[a],vectors[b],grouped,bootstrap=True)['ci95']
                run.require(np.allclose(ci,expected['source_group_ci95'],atol=1e-14,rtol=0),'Source-group interval drift')
        av=[vectors[f'expanded_s{s}'] for s in (1,2,3)];bv=[vectors[f'{arm}_s{s}'] for s in (1,2,3)]
        replay=run.mean_seed_comparison(av,bv,entries);expected=result['seed_mean_comparisons']['expanded vs '+arm]
        for field in ('delta','ci95'):run.require(np.allclose(replay[field],expected[field],atol=1e-14,rtol=0),'Seed mean replay drift')
        if cohort=='new_reserve':
            ci=run.mean_seed_comparison(av,bv,entries,'source_group')['ci95']
            run.require(np.allclose(ci,expected['source_group_ci95'],atol=1e-14,rtol=0),'Seed mean group interval drift')
    return dict(queries=len(entries),models=len(frozen['models']),predictions_replayed=len(entries)*len(frozen['models']),
        coordinates_labels_sha_verified=True,metrics_and_cluster_intervals_replayed=True,
        scope='No model inference or fitting; original state-array authentication was performed during original fitting/evaluation, not repeated in this completion-only repair.')


def main():
    out=run.OUT
    if (out/'completion.json').exists():
        c=run.read(out/'completion.json');run.require(c.get('complete'),'Incomplete completion record')
        for filename,expected in c['artifact_sha256'].items():run.require(run.sha(filename)==expected,'Completion drift')
        print('Already complete; completion artifacts verified.');return
    run.register();frozen=run.frozen_models();run.base.legacy_hashes()
    for filename,expected in run.read(run.base.V13/'frozen.json')['files'].items():
        run.require(run.sha(filename)==expected,'Legacy V13 seal drift')
    repair=out/'repairs/finalization_20261104';repair.mkdir(parents=True,exist_ok=True)
    sources=[Path(__file__),ROOT/'tests/test_finalize_quality_readout_v15_3k.py']
    source_sha={str(p.relative_to(ROOT)):run.sha(p) for p in sources}
    for p in sources:
        target=repair/p.name
        if target.exists():run.require(run.sha(target)==run.sha(p),'Repair source archive changed')
        else:shutil.copyfile(p,target)
    prior=run.read(out/'status.json')
    if not (repair/'original_failure_status.json').exists():run.write(repair/'original_failure_status.json',prior)
    paths=[out/'protocol.json',out/'models/frozen.json',out/'features/new_train/index.json',out/'features/new_reserve/index.json']
    for cohort,count in [('new_train',2100),('new_reserve',300)]:
        ix=run.read(out/f'features/{cohort}/index.json');manifest=out/f'data/{cohort}.jsonl'
        run.require(ix.get('complete') is True and len(ix['cases'])==count and ix['n']==count,'Incomplete feature index')
        run.require(ix['manifest_sha256']==run.sha(manifest) and ix['protocol_sha256']==run.sha(out/'protocol.json'),'Feature index provenance drift')
        cases=run.base.rows(manifest);lookup={c['sample_id']:c for c in cases}
        run.require({e['sample_id'] for e in ix['cases']}==set(lookup),'Feature membership drift')
        for e in ix['cases']:run.require(e['case_identity']==run.digest(lookup[e['sample_id']]),'Feature row drift')
    paths += [out/f'evaluation/{c}/{f}.json' for c in ('old_val','new_reserve') for f in ('results','details')]
    before={str(p):run.sha(p) for p in paths}
    audits={}
    for cohort in ('old_val','new_reserve'):
        print('Auditing '+cohort,flush=True);audits[cohort]=audit_cohort(cohort,frozen)
    run.register()
    run.require(all(run.sha(p)==h for p,h in before.items()),'Scientific artifacts changed during completion repair')
    receipt=dict(complete=True,created_at_utc=datetime.now(timezone.utc).isoformat(),source_sha256=source_sha,
        original_source_identity=run.source_identity(),protocol_sha256=run.sha(out/'protocol.json'),
        root_cause='Original completion guard applied dict.get to list-valued evaluation details.json after both evaluations had finished.',
        change='Standalone completion-only schema guard and cached-prediction audit; sealed scientific source, protocol, models, features, predictions and evaluation JSON files unchanged.',
        original_failure_status_path=str(repair/'original_failure_status.json'),original_failure_status_sha256=run.sha(repair/'original_failure_status.json'),
        audits=audits,original_artifact_sha256=before,retraining_or_feature_extraction=False)
    run.write(out/'checks/completion_repair.json',receipt)
    paths.append(out/'checks/completion_repair.json')
    run.write(out/'completion.json',dict(complete=True,completed_at_utc=datetime.now(timezone.utc).isoformat(),
        artifact_sha256={str(p):run.sha(p) for p in paths},source_identity=run.source_identity(),legacy_unchanged=True,
        new_models=7,evaluated_models=9,independent_final_test=False,completion_repair=str(out/'checks/completion_repair.json')))
    run.status('complete',new_models=7,evaluated_models=9,old_val_queries=172,new_development_queries=300,completion_audit_repaired=True)
    run.write(out/'failure_resolved.json',dict(resolved=True,original_failure_preserved=True,completion_sha256=run.sha(out/'completion.json'),repair_receipt_sha256=run.sha(out/'checks/completion_repair.json')))


if __name__=='__main__':main()
