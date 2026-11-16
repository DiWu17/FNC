"""Declared mid-run, before any outcome scoring; completed-output sensitivity only."""
import hashlib
import json
from datetime import datetime,timezone
from pathlib import Path
import re
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from clickagent.public_evaluation import point_in_geometry
OUT=ROOT/'runs/visual_feedback_v23'


def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def compatible_point(output):
    if output['valid'] or not output['terminated_by_eos']:return output['point'],False
    text=output['generated_text'].strip();g=output['geometry']
    if text in g['labels']:return list(g['shown'][g['labels'].index(text)]),True
    m=re.fullmatch(r'CLICK\s+(\d{1,4})\s*,\s*(\d{1,4})',text)
    if m and all(0<=int(v)<=1000 for v in m.groups()):
        return [round(int(v)*(s-1)/1000) for v,s in zip(m.groups(),g['original_size'])],True
    return output['point'],False


def register():
    assert not (OUT/'results.json').exists(), 'Sensitivity declared before any outcome scoring'
    assert not (OUT/'parser_sensitivity_registration.json').exists()
    g=dict(labels=['A','B'],shown=[[10,20],[30,40]],original_size=[2001,1001])
    base=dict(valid=False,terminated_by_eos=True,point=[1,2],geometry=g)
    assert compatible_point(dict(base,generated_text='B'))==([30,40],True)
    assert compatible_point(dict(base,generated_text='CLICK 500, 250'))==([1000,250],True)
    assert compatible_point(dict(base,generated_text='CLICK 500, 250',terminated_by_eos=False))==([1,2],False)
    assert compatible_point(dict(base,generated_text='CLICK 1001, 250'))==([1,2],False)
    assert compatible_point(dict(base,generated_text='B extra'))==([1,2],False)
    write(OUT/'parser_sensitivity_registration.json',dict(declared_at_utc=datetime.now(timezone.utc).isoformat(),
        stage='Mid-inference after9 format deviations, before any outcome labels/scores were inspected in this experiment.',
        primary_unchanged=True,classification='Supplementary compatibility analysis,not original preregistered primary policy.',
        rule='For completed-EOS outputs only, accept a bare available candidate letter as KEEP letter,or exact CLICK integer, integer with comma as CLICK integer integer. All other rules unchanged. No salvage of truncated or explanatory outputs.',
        source_sha256=sha(__file__),synthetic_checks_passed=5))


def analyze():
    assert read(OUT/'completion.json')['complete']
    registration=read(OUT/'parser_sensitivity_registration.json');assert registration['source_sha256']==sha(__file__)
    from run_visual_feedback_v23 import bootstrap
    cases=read(OUT/'data/public.json');labels=read(OUT/'data/labels.json');pred=read(OUT/'predictions.json')['rows']
    lookup={};modified=[];cached={}
    for r in pred:
        p=r['point'];changed=False
        if r['task_id'] and r['arm']!='marked_verify':
            tid=r['task_id']
            if tid not in cached:cached[tid]=compatible_point(read(OUT/f'calls/{tid}.json')['output'])
            p,changed=cached[tid]
        c=next(c for c in cases if c['sample_id']==r['sample_id'])
        success=bool(p is not None and point_in_geometry([v/s for v,s in zip(p,c['image_size'])],labels[c['sample_id']]))
        lookup[(r['model'],r['arm'],r['sample_id'])]=success
        if changed:modified.append(dict(sample_id=r['sample_id'],model=r['model'],arm=r['arm'],strict_point=r['point'],compatible_point=p,strict_hit=r['hit'],compatible_hit=success,task_id=r['task_id']))
    results={}
    for co in ['new_development','old_val','all']:
        selected=[c for c in cases if co=='all' or c['cohort']==co];out={}
        for f in ['original3k','full']:
            out[f]={}
            base=np.asarray([[lookup[(f'{f}_s{s}','baseline',c['sample_id'])] for c in selected] for s in [1,2,3]],float)
            for arm in ['baseline','local_relook','marked_verify','marked_refine']:
                v=np.asarray([[lookup[(f'{f}_s{s}',arm,c['sample_id'])] for c in selected] for s in [1,2,3]],float)
                out[f][arm]=dict(accuracy_percent=100*float(v.mean()),seed_hits=v.sum(1).astype(int).tolist(),
                    vs_baseline_descriptive95=bootstrap(v-base,selected,.95))
        results[co]=dict(n=len(selected),methods=out)
    write(OUT/'parser_sensitivity.json',dict(complete=True,registration=registration,primary_unchanged=True,
        cohorts=results,modified_logical_outputs=modified,unique_salvaged_calls=sum(changed for _,changed in cached.values()),
        saved_primary_results_sha256=sha(OUT/'results.json')))
    print(json.dumps(dict(complete=True,unique_salvaged_calls=sum(changed for _,changed in cached.values()),new72=results['new_development']),ensure_ascii=False),flush=True)


if __name__=='__main__':
    if '--register' in sys.argv:register()
    else:analyze()
