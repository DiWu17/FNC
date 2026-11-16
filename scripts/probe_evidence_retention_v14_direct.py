"""Explicitly posthoc direct-quality readout diagnosis, never a primary arm.

Motivation: the prespecified frozen old-head branch did not convert the modest
retention improvement to Top1 hits. Probe the already-frozen new quality scores
on (a) the unchanged old K16 pool and (b) the decoded union. This changes the
selector, so it cannot replace the fixed-downstream experiment's conclusion.
No training, extra model forward, new coordinate or GUI-score consumption.
"""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'src/clickagent').is_dir() and (p/'runs/evidence_retention_v14').is_dir())
RUN=ROOT/'runs/evidence_retention_v14'


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    out=RUN/'analysis/direct_quality_readout';out.mkdir(parents=True,exist_ok=True)
    spec=dict(posthoc=True,created_at_utc=datetime.now(timezone.utc).isoformat(),
        motivation='Prespecified fixed old internal Top1: baseline75/75, aligned pools72/73 on development val172. Diagnose whether the new quality scores themselves rank useful points.',
        arms=['rank_s1','rank_s2','aligned_s1','aligned_s2','permuted_s1','permuted_s2'],
        pools=['unchanged sealed K16 coordinates','all previously decoded coordinates'],
        inference='Forced argmax of existing saved quality scores. No training or new forward. Same actual old-pool size in first comparison; union comparison scores more points and is separate.',
        interpretation='Exploratory readout change, not an isolated retention effect. Cannot replace experiment3 main fixed-downstream results or establish independent generalization.',
        baseline_CPU_result_sha256=sha(RUN/'experiment3/cpu_top1_interim.json'))
    path=out/'protocol.json'
    if not path.exists():path.write_text(json.dumps(spec,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    else:
        old=read(path);spec['created_at_utc']=old['created_at_utc'];assert old==spec
    entries=[e for e in read(RUN/'base/index.json')['cases'] if e['split']=='val']
    methods=spec['arms'];vectors={p:{n:[] for n in methods} for p in ['old_pool','decoded_union']};details=[]
    for e in entries:
        with np.load(e['path'],allow_pickle=False) as z:old=z['baseline'].copy();points=z['points'].copy()
        labels=np.load(e['labels_path'],allow_pickle=False);row=dict(sample_id=e['sample_id'],methods={})
        for name in methods:
            checkpoint=read(RUN/f'models/{name}.json')['checkpoint_sha256']
            path=RUN/f'predictions/{name}/val/{hashlib.sha256(e["sample_id"].encode()).hexdigest()[:24]}.npz'
            with np.load(path,allow_pickle=False) as z:
                assert str(z['checkpoint_sha256'])==checkpoint and str(z['base_sha256'])==e['sha256']
                scores=z['scores'].copy()
            assert len(scores)==len(points) and np.isfinite(scores).all()
            for pool,ix in [('old_pool',np.sort(old)),('decoded_union',np.arange(len(points)))]:
                pick=int(ix[int(scores[ix].argmax())]) if len(ix) else None
                yes=bool(pick is not None and labels[pick])
                vectors[pool][name].append(yes)
                row['methods'][pool+'__'+name]=dict(point=points[pick].tolist() if pick is not None else None,hit=yes)
        details.append(row)
    comparisons={}
    for pool,values in vectors.items():
        comparisons[pool]={}
        for s in [1,2]:
            a=np.asarray(values[f'aligned_s{s}'])
            for ref in [f'rank_s{s}',f'permuted_s{s}']:
                b=np.asarray(values[ref]);comparisons[pool][f'aligned_s{s} vs '+ref]=dict(wins=int((a&~b).sum()),losses=int((b&~a).sum()),delta_hits=int(a.sum()-b.sum()))
    result=dict(complete=True,posthoc=True,n=172,hits={p:{n:sum(v) for n,v in values.items()} for p,values in vectors.items()},
        comparisons=comparisons,scope=spec['interpretation'],changes_original_experiments=False)
    (out/'results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (out/'details.json').write_text(json.dumps(details,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (out/'source.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
