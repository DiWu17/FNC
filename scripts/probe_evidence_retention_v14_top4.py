"""Posthoc plan/readout probe; preserves all three primary experiments.

Plan before reading GUI outcomes, evaluate only after their completion. Uses
already-frozen quality scores and shared GUI scores, with no new forwards.
"""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'src/clickagent').is_dir() and (p/'runs/evidence_retention_v14').is_dir())
RUN=ROOT/'runs/evidence_retention_v14'


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def key(sid):return hashlib.sha256(sid.encode()).hexdigest()[:24]


def main():
    out=RUN/'analysis/quality_top4_gui';out.mkdir(parents=True,exist_ok=True)
    methods=[f'{v}_s{s}' for v in ['rank','aligned','permuted'] for s in [1,2]]
    plan=dict(posthoc=True,created_at_utc=datetime.now(timezone.utc).isoformat(),
        motivation='Direct frozen quality readout on unchanged oldK16 gave aligned82/82 vs rank76/76 and oldhead75/75. Diagnose whether the22k readout can prescreen the shared GUI verifier.',
        endpoints=['quality Top4 on unchanged oldK16, all6 frozen methods','quality Top4 from decoded union, rank/aligned S1/S2 only; every such point must already belong to experiment3 GUI jobs'],
        control='Same4-point verifier budget as old internal-head Top4; no new model fitting, policy changes, model forwards or coordinate generation.',
        timing='Plan written during pending GUI execution, after CPU-only interim outcomes; no GUI-hit outcomes used to choose this plan.',
        scope='Exploratory additional readout diagnosis. Does not replace prespecified fixed-downstream results or constitute independent confirmation.',
        models={m:sha(RUN/f'models/{m}.pt') for m in methods})
    pp=out/'plan.json'
    if pp.exists():old=read(pp);plan['created_at_utc']=old['created_at_utc'];assert old==plan
    else:pp.write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (out/'source.py').write_bytes(Path(__file__).read_bytes())
    complete=RUN/'experiment3/gui/completion.json'
    if not complete.exists():print(json.dumps(dict(planned=True,gui_pending=True)));return
    assert read(complete)['complete']
    import sys
    sys.path.insert(0,str(ROOT/'src'))
    from clickagent.public_evaluation import point_in_geometry
    cases={c['sample_id']:c for c in (json.loads(l) for l in (ROOT/'runs/multiregion_v13/data/val.jsonl').read_text(encoding='utf-8').splitlines())}
    entries=[e for e in read(RUN/'base/index.json')['cases'] if e['split']=='val']
    vectors={};oracle={};details=[]
    for e in entries:
        sid=key(e['sample_id']);case=cases[e['sample_id']]
        with np.load(e['path'],allow_pickle=False) as z:points=z['points'].copy();old=np.sort(z['baseline'])
        labels=np.load(e['labels_path'],allow_pickle=False)
        gui=read(RUN/f'experiment3/gui/cases/{sid}.json');assert gui['complete']
        gs={tuple(r['point']):r['log_odds'] for r in gui['scores']};row={}
        for method in methods:
            with np.load(RUN/f'predictions/{method}/val/{sid}.npz',allow_pickle=False) as z:scores=z['scores'].copy()
            choices=[('old_pool',old)]
            if not method.startswith('permuted'):choices.append(('decoded_union',np.arange(len(points))))
            for pool,ix in choices:
                selected=np.sort(ix[np.argsort(-scores[ix],kind='stable')[:4]])
                candidates=points[selected].tolist();assert all(tuple(p) in gs for p in candidates)
                chosen=max(candidates,key=lambda p:gs[tuple(p)]) if candidates else None
                yes=bool(chosen is not None and point_in_geometry([chosen[0]/case['image_size'][0],chosen[1]/case['image_size'][1]],case['target_geometry']))
                name=pool+'__'+method;reachable=bool(labels[selected].any())
                vectors.setdefault(name,[]).append(yes);oracle.setdefault(name,[]).append(reachable)
                row[name]=dict(point=chosen,hit=yes,top4_coverage=reachable)
        details.append(dict(sample_id=e['sample_id'],methods=row))
    result=dict(complete=True,posthoc=True,n=172,
        methods={n:dict(hits=sum(v),top4_coverage=sum(oracle[n]),n=172) for n,v in vectors.items()},
        scope=plan['scope'],changes_original_experiments=False,additional_model_forward_calls=0)
    (out/'results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (out/'details.json').write_text(json.dumps(details,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
