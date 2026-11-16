"""Prospective secondary scaling contrasts, run after the registered fresh test."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'runs/scaling_effects_v18'
FULL = ROOT / 'runs/full_scaling_v18'
PARENT = ROOT / 'runs/mvp_confirmation_v18'
JOINT = PARENT / 'coordinate_addendum'
CONTRASTS = {
    'full_vs_matched3k': ('full_union', 'matched3k_union'),
    'matched3k_vs_original3k': ('matched3k_union', 'original3k_union'),
    'full_vs_original3k': ('full_union', 'original3k_union'),
}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    temp.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def status(phase, **fields):
    value = dict(phase=phase, complete=phase == 'complete',
                 updated_at_utc=datetime.now(timezone.utc).isoformat(), **fields)
    write(OUT / 'status.json', value)
    print(json.dumps(value, ensure_ascii=False), flush=True)


def register():
    path = OUT / 'protocol.json'
    if not path.exists():
        assert not list((PARENT / 'predictions').rglob('*.json')), 'Register secondary contrasts before any heldout predictions'
        files = [Path(__file__), FULL/'protocol.json', PARENT/'protocol.json', JOINT/'protocol.json',
                 PARENT/'data/heldout.jsonl', FULL/'data/development_indices.json']
        write(path, dict(registered_at_utc=datetime.now(timezone.utc).isoformat(),
                         hashes={str(p):sha(p) for p in files}, contrasts=CONTRASTS,
                         scope='Secondary descriptive contrasts only; no new model, inference, policy selection, or primary MVP claim',
                         uncertainty='2000 paired source-stratified RGB-cluster bootstrap draws, seed20261401, descriptive95% intervals; fixed3 fits averaged within each screenshot',
                         interpretations=dict(full_vs_matched3k='Data plus source mixture change at the same13376-update budget; not pure quantity effect',
                                              matched3k_vs_original3k='Additional optimizer updates on the same2973 rows:13376 vs2976',
                                              full_vs_original3k='Combined data/mixture and update-budget change'),
                         fresh_scope='The reserved280 screenshot subset only; old172/new300 development reported separately',
                         primary_MVP_family='Existing coordinate_addendum combined three-comparison family remains unchanged'))
    spec = read(path)
    for p, digest in spec['hashes'].items():
        assert sha(p) == digest, p
    assert spec['contrasts'] == {name:list(pair) for name,pair in CONTRASTS.items()}
    return spec


def contrasts(values, entries):
    strata = {}
    for i, entry in enumerate(entries):
        strata.setdefault(entry['source'], {}).setdefault(entry['group'], []).append(i)
    groups = [list(g.values()) for g in strata.values()]
    rng = np.random.default_rng(20261401)
    draws = [[i for clusters in groups for j in rng.integers(0,len(clusters),len(clusters)) for i in clusters[j]]
             for _ in range(2000)]
    result = {}
    for name, (arm, reference) in CONTRASTS.items():
        a, b = np.asarray(values[arm], float), np.asarray(values[reference], float)
        assert a.shape == b.shape == (3,len(entries))
        delta = (a-b).mean(0)
        ci = np.quantile([delta[ix].mean() for ix in draws], [.025,.975])
        result[name] = dict(delta_pp=100*float(delta.mean()), ci95_pp=(100*ci).tolist(),
                            seed_deltas_pp=(100*(a-b).mean(1)).tolist(),
                            by_source={source:100*float(delta[[i for i,e in enumerate(entries) if e['source']==source]].mean())
                                       for source in sorted(strata)}, descriptive_only=True)
    return result


def analyze():
    provenance = {}
    for folder in [FULL, PARENT, JOINT]:
        path = folder / 'completion.json'
        receipt = read(path)
        assert receipt['complete']
        provenance[str(path)] = sha(path)
        for p, digest in receipt['artifact_sha256'].items():
            assert sha(p) == digest, p
    index = read(FULL/'data/development_indices.json')
    outputs = {}
    for cohort, entries in index.items():
        path = FULL/f'evaluation/{cohort}.json'
        source = read(path)
        provenance[str(path)] = sha(path)
        values = {arm:np.asarray([source['hit_vectors'][f'{name}_s{s}'] for s in [1,2,3]],float)
                  for arm,name in [('full_union','full'),('matched3k_union','matched3k'),('original3k_union','original3k')]}
        outputs[cohort] = dict(n=len(entries), scope='Reused development', contrasts=contrasts(values,entries))
    cases = [json.loads(line) for line in (PARENT/'data/heldout.jsonl').read_text(encoding='utf-8').splitlines()]
    path = PARENT/'predictions/ours.json'
    own = {r['sample_id']:r for r in read(path)['rows']}
    provenance[str(path)] = sha(path)
    assert len(cases) == 280 and set(own) == {c['sample_id'] for c in cases}
    values = {arm:np.asarray([own[c['sample_id']]['hits'][arm] for c in cases],float).T
              for arm in ['full_union','matched3k_union','original3k_union']}
    entries = [dict(source=c['dataset'], group=c['pixel_sha256']) for c in cases]
    outputs['fresh_280'] = dict(n=280, scope='Reserved fresh screenshot subset', contrasts=contrasts(values,entries))
    lines = ['V18 数据扩展与训练步数：预先登记的次要描述性比较。',
             '所有差值均为固定三个种子的平均准确率差，单位为百分点；95%区间不承担多重比较后的主结论。',
             'full_vs_matched3k：相同步数下的数据及来源混合变化；不能单独归因为条数。',
             'matched3k_vs_original3k：相同2973条数据，13376步对2976步。',
             'full_vs_original3k：数据/来源与步数同时变化。']
    for cohort, result in outputs.items():
        lines += ['', f'{cohort}, n={result["n"]}']
        for name, row in result['contrasts'].items():
            lo,hi = row['ci95_pp']
            lines.append(f'{name}: {row["delta_pp"]:+.2f} pp; 描述性95%区间[{lo:+.2f},{hi:+.2f}]; 三种子差值{row["seed_deltas_pp"]}')
    write(OUT/'results.json', dict(complete=True, cohorts=outputs, primary_MVP_results=str(JOINT/'combined_results.json'),
                                   mainline_unchanged=True, descriptive_only=True))
    (OUT/'results.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    register()
    paths = [OUT/'protocol.json',OUT/'results.json',OUT/'results.txt']
    write(OUT/'completion.json',dict(complete=True,completed_at_utc=datetime.now(timezone.utc).isoformat(),
                                    input_sha256=provenance, artifact_sha256={str(p):sha(p) for p in paths}))
    status('complete')
    print('\n'.join(lines),flush=True)


def main():
    register()
    while not (JOINT/'completion.json').exists():
        for folder in [FULL,PARENT,JOINT]:
            path=folder/'status.json'
            if path.exists() and read(path).get('phase')=='failed':
                raise RuntimeError('Dependency failed: '+str(folder))
        status('waiting_for_joint_fresh_analysis')
        time.sleep(50)
    analyze()


if __name__=='__main__':
    try:
        main()
    except BaseException:
        status('failed',error=traceback.format_exc())
        raise
