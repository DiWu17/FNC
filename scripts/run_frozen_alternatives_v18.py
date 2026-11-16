"""CPU-only development comparisons with the adopted scorer kept immutable."""
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import numpy as np
from clickagent.frozen_selection_v18 import METHODS, select

OUT = ROOT / 'runs/frozen_alternatives_v18'
MAIN = ROOT / 'runs/quality_readout_v15_3k'
FULL = ROOT / 'runs/full_scaling_v18'


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(p, value):
    p.parent.mkdir(parents=True, exist_ok=True)
    temp = p.with_suffix(p.suffix+'.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    temp.replace(p)


def status(phase, **fields):
    row = dict(phase=phase, complete=phase == 'complete', pid=__import__('os').getpid(),
               updated_at_utc=datetime.now(timezone.utc).isoformat(), **fields)
    write(OUT / 'status.json', row)
    print(json.dumps(row, ensure_ascii=False), flush=True)


def register():
    path = OUT / 'protocol.json'
    if path.exists():
        spec = read(path)
        for p, expected in spec['input_hashes'].items():
            assert sha(p) == expected, p
        return spec
    files = [Path(__file__), ROOT / 'src/clickagent/frozen_selection_v18.py',
             ROOT / 'tests/test_frozen_selection_v18.py', FULL / 'data/development_indices.json',
             MAIN / 'models/frozen.json', MAIN / 'completion.json',
             ROOT / 'runs/multiregion_v13/data/val.jsonl', MAIN / 'data/new_reserve.jsonl']
    spec = dict(registered_at_utc=datetime.now(timezone.utc).isoformat(),
        input_hashes={str(p): sha(p) for p in files}, methods=list(METHODS),
        reference='union: frozen existing scorer argmax over every original generated point',
        rules='Distinct-view Gaussian proximity support with radius.015 of image diagonal, top8 proposer-ranked points per view. RRF60. Fixed quality-percentile plus support/4 weights.25,.5,1; top4-consensus selector.',
        primary='New300 historically used development queries; conditional3-fixed-seed mean, paired source-stratified RGB-cluster bootstrap2000;99% intervals for5 alternatives versus union.',
        guard='Old172 development reported separately, not pooled with new300.',
        inference='Only cached scores/points/proposal features; no model inference, additional pixels, scorer fitting or change to any stored score.',
        scope='Development exploration. No independent confirmation or policy adoption based on the consumed V16/V17 test screenshots.',
        seeds=[1, 2, 3], mainline_unchanged=True, gpu_used=False, html=False)
    write(path, spec); status('registered', methods=len(METHODS))
    return spec


def comparison(a, b, entries):
    diff = np.mean(np.asarray(a, float)-np.asarray(b, float), axis=0)
    strata = defaultdict(lambda: defaultdict(list))
    for i, e in enumerate(entries):
        strata[e['source']][e['group']].append(i)
    clusters = [list(g.values()) for g in strata.values()]
    rng = np.random.default_rng(20261105); values = []
    for _ in range(2000):
        ids = [i for groups in clusters for j in rng.integers(0, len(groups), len(groups)) for i in groups[j]]
        values.append(float(diff[ids].mean()))
    return dict(delta_pp=100*float(diff.mean()), ci95_pp=(100*np.quantile(values, [.025, .975])).tolist(),
                family5_ci99_pp=(100*np.quantile(values, [.005, .995])).tolist())


def main():
    spec = register(); indices = read(FULL / 'data/development_indices.json')
    frozen = read(MAIN / 'models/frozen.json')['models']
    public = {}
    for p in [ROOT / 'runs/multiregion_v13/data/val.jsonl', MAIN / 'data/new_reserve.jsonl']:
        public.update({c['sample_id']: c for c in (json.loads(s) for s in p.read_text(encoding='utf-8').splitlines())})
    results = {}; provenance = {}
    for cohort, entries in indices.items():
        status('evaluating_'+cohort, completed=0, total=len(entries))
        vectors = {m: [[], [], []] for m in METHODS}; picks = []
        for i, e in enumerate(entries, 1):
            for field, h in [('path', 'sha256'), ('labels_path', 'labels_sha256')]:
                assert sha(e[field]) == e[h]
                provenance[e[field]] = e[h]
            with np.load(e['path'], allow_pickle=False) as z:
                points, base, baseline = [z[k].copy() for k in ['points', 'base', 'baseline']]
            # Candidate annotations are accessed only after every public-only selection.
            selections = []
            for s in [1, 2, 3]:
                legacy_cohort = 'old_val' if cohort == 'old_val' else 'new_reserve'
                p = MAIN / f'predictions/{legacy_cohort}/expanded_s{s}/{Path(e["path"]).stem}.npz'
                with np.load(p, allow_pickle=False) as z:
                    assert str(z['checkpoint_sha256']) == frozen[f'expanded_s{s}']['sha256']
                    assert str(z['base_sha256']) == e['sha256']
                    scores = z['scores'].copy()
                provenance[str(p)] = sha(p)
                selections.append(select(points, base, baseline, scores, public[e['sample_id']]['image_size']))
            labels = np.load(e['labels_path'], allow_pickle=False)
            for s, selected in enumerate(selections):
                for method, pick in selected.items():
                    vectors[method][s].append(bool(pick is not None and labels[pick]))
            picks.append(dict(sample_id=e['sample_id'], selected_indices=selections))
            if i % 50 == 0 or i == len(entries):
                status('evaluating_'+cohort, completed=i, total=len(entries))
        methods = {m: dict(seed_hits=np.asarray(v).sum(1).tolist(),
                    mean_accuracy_percent=100*float(np.mean(v))) for m, v in vectors.items()}
        comparisons = {m: comparison(vectors[m], vectors['union'], entries) for m in METHODS if m not in ['K16', 'union']}
        by_source = {}
        for source in sorted({e['source'] for e in entries}):
            ids = [i for i, e in enumerate(entries) if e['source'] == source]
            by_source[source] = dict(n=len(ids), methods={m:100*float(np.asarray(v)[:, ids].mean()) for m,v in vectors.items()})
        result = dict(complete=True, n=len(entries), unique_rgb=len({e['group'] for e in entries}),
            methods=methods, comparisons=comparisons, by_source=by_source,
            sample_ids=[e['sample_id'] for e in entries], hit_vectors=vectors,
            selected_indices=picks, scope=spec['scope'])
        write(OUT / f'evaluation/{cohort}.json', result); results[cohort] = result
    for name, expected in spec['input_hashes'].items():
        assert sha(name) == expected
    lines = ['冻结原3000例评分器：只改变候选选择规则；CPU缓存开发实验，不是新的独立测试。',
             '新300开发集为主；旧172开发集单列；所有三个种子均报告。']
    for cohort, result in results.items():
        lines.append('\n'+cohort)
        for method, row in result['methods'].items():
            line = f'{method}: {row["mean_accuracy_percent"]:.2f}%; 三种子命中{row["seed_hits"]}'
            if method in result['comparisons']:
                c = result['comparisons'][method]
                line += f'; 相对完整候选argmax {c["delta_pp"]:+.2f}百分点, 五项调整99%区间{c["family5_ci99_pp"]}'
            lines.append(line)
    lines.append('\n本轮不使用V16/V17复用测试进行新策略选择；开发收益仍须冻结策略后独立确认。')
    path = OUT / 'results.txt'; path.write_text('\n'.join(lines)+'\n', encoding='utf-8')
    write(OUT / 'completion.json', dict(complete=True, completed_at_utc=datetime.now(timezone.utc).isoformat(),
        inputs_sha256=provenance, artifact_sha256={str(p): sha(p) for p in [OUT/'protocol.json', path]+[OUT/f'evaluation/{c}.json' for c in results]},
        adopted_scorer_unchanged=True, no_training_or_model_inference=True))
    status('complete', methods=len(METHODS), cohorts={c:r['n'] for c,r in results.items()})
    print('\n'.join(lines), flush=True)


if __name__ == '__main__':
    try:
        main()
    except BaseException:
        import traceback
        status('failed', error=traceback.format_exc()); raise
