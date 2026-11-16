"""Retrospective full-scale supplement on V16's fixed 300-query cohort.

Register the computation before CPU scoring. Reuse exact, SHA-sealed V2
ScreenSpot-Pro predictions; score the Venus V16 label-free feature cache.
Only ``analyze`` may open the already historically observed labels, and only
after all 300 cases pass the prediction seal gate. This is descriptive, not a
new independent confirmation.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
import numpy as np
import run_paper_scale_eval_v1 as engine
from clickagent.contribution_ablation import VARIANTS
from clickagent.public_evaluation import point_in_geometry

V16 = ROOT / 'runs/contribution_v16'
V2 = ROOT / 'runs/paper_scale_ablation_v2/eval/full'
V17 = ROOT / 'runs/advantage_validation_v17/inference'
OUT = ROOT / 'runs/paper_full_300_supplement_v1'
PROTOCOL = OUT / 'protocol.json'
SEAL = OUT / 'predictions.json'
RECEIPT = OUT / 'public_completion.json'
MANIFEST = V16 / 'data/independent_test.jsonl'
INDEX = V16 / 'independent/index.json'
V2_INDEX = V2 / 'features/index.json'
V2_SEAL = V2 / 'predictions.json'
MVP = V17 / 'mvp_p262144_all/accuracy/results.json'
GUI = V17 / 'ours_gui_p262144_all/accuracy/results.json'
MODEL_REGISTRY = ROOT / 'runs/paper_scale_ablation_v1/models/frozen.json'
REPLICATES = 10000
BOOT_SEED = 20261113
SEEDS = (1, 2, 3)
METHODS = tuple(f'full_{v}_s{s}' for v in VARIANTS for s in SEEDS)


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def require(test, message):
    if not test:
        raise ValueError(message)


def write_once(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        require(read(path) == data, 'Existing output differs: ' + str(path))
        return
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temp.replace(path)


def public_inventory():
    v16 = read(INDEX)
    v2 = read(V2_INDEX)
    a = v16['cases']
    b = {e['sample_id']: e for e in v2['cases']}
    require(v16['complete'] and v16['n'] == 300 and v2['complete'] and
            v2['n'] == 1581 and len(a) == 300 and len(b) == 1581,
            'Cohort inventory changed')
    require(len({e['sample_id'] for e in a}) == 300, 'Duplicate V16 sample IDs')
    require(sum(e['source'] == 'ScreenSpot-Pro' for e in a) == 150 and
            sum(e['source'] == 'VenusBench-GD' for e in a) == 150,
            'V16 150+150 cohort changed')
    for e in a:
        if e['source'] == 'ScreenSpot-Pro':
            ref = b[e['sample_id']]
            for field in ('sha256', 'state_sha256', 'mask_sha256'):
                require(e[field] == ref[field], 'V16/V2 feature mismatch: ' + e['sample_id'])
            for field in ('path', 'state_path', 'mask_path'):
                require(sha(e[field]) == e[field.replace('_path', '_sha256') if field != 'path' else 'sha256'],
                        'V16 feature bytes changed: ' + e['sample_id'])
        else:
            for path_field, hash_field in (('path', 'sha256'), ('state_path', 'state_sha256'),
                                           ('mask_path', 'mask_sha256')):
                require(sha(e[path_field]) == e[hash_field],
                        'Venus feature bytes changed: ' + e['sample_id'])
    return a


def model_table():
    table = engine.models(False)
    require(all(name in table for name in METHODS) and len(METHODS) == 18,
            'The six full variants x three seeds are not frozen')
    return {name: table[name] for name in METHODS}


def register():
    entries = public_inventory()
    models = model_table()
    v2_seal = read(V2_SEAL)
    require(v2_seal['complete'] and v2_seal['n'] == 1581 and
            v2_seal['model_hashes'] == read(V2 / 'protocol.json')['model_hashes'],
            'V2 full prediction seal changed')
    spec = dict(
        complete=True,
        scope='Retrospective V16/V17 300-query supplement; historical labels/results already seen',
        cohort='fixed V16 ScreenSpot-Pro150 + VenusBench-GD150', n=300,
        sample_ids=[e['sample_id'] for e in entries],
        sample_sources=[e['source'] for e in entries],
        input_sha256={str(p): sha(p) for p in
                      (INDEX, MANIFEST, V2_INDEX, V2_SEAL, MODEL_REGISTRY, MVP, GUI)},
        model_paths={name: models[name]['path'] for name in METHODS},
        model_sha256={name: models[name]['sha256'] for name in METHODS},
        inference='Original V16 K16 candidate indices; forced Top1; stable score sort with canonical union index tie',
        ssp_prediction_reuse='V2 full1581 sealed case files, with exact V16/V2 base/state/mask SHA parity',
        venus_prediction_source='V16 label-free candidate/base/state/mask cache; CPU score_record',
        metric='Original predicted point inside V16 target geometry; null point is wrong',
        analysis=dict(
            family='descriptive, previously observed 300; no new independent confirmation',
            subsets=['all300', 'ScreenSpot-Pro150', 'VenusBench-GD150'],
            primary_differences=['full_aligned-minus-full_no_evidence',
                                 'full_aligned-minus-full_spatial_shuffle',
                                 'full_aligned-minus-V17_MVP',
                                 'full_aligned-minus-V17_GUI_Actor_K16'],
            seed_aggregation='mean of the fixed three matched seeds, then paired query delta',
            bootstrap_unit='RGB screenshot cluster, stratified by dataset',
            bootstrap_replicates=REPLICATES, bootstrap_seed=BOOT_SEED,
            intervals=['95% marginal', '98.333% three-comparison-family reference'],
            cluster_coverage='K16 oracle and MVP three-proposal oracle, fixed sets',
        ),
        original_sources_unchanged=True,
    )
    write_once(PROTOCOL, spec)
    print('Registered:', PROTOCOL, 'sha256', sha(PROTOCOL), flush=True)


def verify_registered():
    p = read(PROTOCOL)
    require(p['complete'] and p['n'] == 300 and len(p['sample_ids']) == 300,
            'Protocol missing or malformed')
    for path, expected in p['input_sha256'].items():
        require(sha(path) == expected, 'Input changed since registration: ' + path)
    entries = public_inventory()
    require([e['sample_id'] for e in entries] == p['sample_ids'], 'Registered ID order changed')
    table = model_table()
    require({name: e['sha256'] for name, e in table.items()} == p['model_sha256'],
            'Registered full models changed')
    return p, entries, table


def venus_record(entry):
    with np.load(entry['path'], allow_pickle=False) as z:
        record = {k: z[k].copy() for k in ('points', 'base', 'active', 'baseline')}
    record.update(sample_id=entry['sample_id'], n=entry['n'],
                  state=np.load(entry['state_path'], mmap_mode='r', allow_pickle=False),
                  mask=np.load(entry['mask_path'], mmap_mode='r', allow_pickle=False))
    require(record['n'] == len(record['points']) and record['base'].shape == (entry['n'], 32)
            and record['state'].shape[0] == entry['n']
            and record['mask'].shape == record['state'].shape[:-1]
            and ((record['baseline'] >= 0) & (record['baseline'] < entry['n'])).all(),
            'Venus feature shape/index invalid: ' + entry['sample_id'])
    return record


def predict():
    p, entries, table = verify_registered()
    v2_seal = read(V2_SEAL)
    loaded = None
    case_hashes = {}
    for ix, e in enumerate(entries, 1):
        sid = e['sample_id']
        case_path = OUT / 'predictions/cases' / (engine.key(sid) + '.json')
        if case_path.exists():
            saved = read(case_path)
            require(saved['sample_id'] == sid and saved['protocol_sha256'] == sha(PROTOCOL),
                    'Supplement cached prediction identity changed: ' + sid)
        elif e['source'] == 'ScreenSpot-Pro':
            original = V2 / 'predictions/cases' / (engine.key(sid) + '.json')
            require(sha(original) == v2_seal['case_hashes'][sid], 'V2 case seal mismatch: ' + sid)
            old = read(original)
            require(old['complete'] and old['sample_id'] == sid and
                    old['feature_sha256'] == e['sha256'] and
                    all(old['model_hashes'][name] == p['model_sha256'][name] for name in METHODS),
                    'V2/V16 prediction-feature-model identity mismatch: ' + sid)
            methods = {name: old['methods'][name] for name in METHODS}
            saved = dict(complete=True, sample_id=sid, source=e['source'],
                         feature_sha256=e['sha256'], protocol_sha256=sha(PROTOCOL),
                         model_sha256=p['model_sha256'], candidate_count=e['n'],
                         K16_count=old['K16_count'], methods=methods,
                         origin='V2 full1581 sealed predictions', origin_sha256=sha(original),
                         labels_opened=False)
            write_once(case_path, saved)
        else:
            if loaded is None:
                loaded = engine.load_model_table(table, 'cpu')
            r = venus_record(e)
            methods = engine.score_record(r, loaded, 'cpu')
            saved = dict(complete=True, sample_id=sid, source=e['source'],
                         feature_sha256=e['sha256'], protocol_sha256=sha(PROTOCOL),
                         model_sha256=p['model_sha256'], candidate_count=e['n'],
                         K16_count=len(r['baseline']), methods=methods,
                         origin='V16 label-free features, CPU scoring', labels_opened=False)
            write_once(case_path, saved)
        require(saved['complete'] and saved['labels_opened'] is False and
                set(saved['methods']) == set(METHODS) and
                saved['model_sha256'] == p['model_sha256'],
                'Incomplete prediction case: ' + sid)
        case_hashes[sid] = sha(case_path)
        if ix % 25 == 0 or ix == len(entries):
            print(f'Predictions {ix}/{len(entries)}', flush=True)
    seal = dict(complete=True, n=300, sample_ids=p['sample_ids'],
                protocol_sha256=sha(PROTOCOL), model_sha256=p['model_sha256'],
                case_sha256=case_hashes, labels_opened=False)
    write_once(SEAL, seal)
    write_once(RECEIPT, dict(complete=True, scope='300 label-free full predictions',
                             predictions_sha256=sha(SEAL), protocol_sha256=sha(PROTOCOL),
                             labels_opened=False))
    print('Predictions sealed:', SEAL, 'sha256', sha(SEAL), flush=True)


def gate():
    p, entries, _ = verify_registered()
    seal, receipt = read(SEAL), read(RECEIPT)
    require(seal['complete'] and seal['n'] == 300 and seal['sample_ids'] == p['sample_ids']
            and seal['protocol_sha256'] == sha(PROTOCOL) and seal['labels_opened'] is False
            and receipt['complete'] and receipt['labels_opened'] is False
            and receipt['predictions_sha256'] == sha(SEAL)
            and len(seal['case_sha256']) == 300,
            'Prediction seal or completion receipt invalid')
    cases = {}
    for e in entries:
        sid = e['sample_id']
        path = OUT / 'predictions/cases' / (engine.key(sid) + '.json')
        require(sha(path) == seal['case_sha256'][sid], 'Per-case prediction SHA changed: ' + sid)
        c = read(path)
        require(c['complete'] and c['labels_opened'] is False and c['sample_id'] == sid
                and c['source'] == e['source'] and c['feature_sha256'] == e['sha256']
                and c['model_sha256'] == p['model_sha256']
                and set(c['methods']) == set(METHODS),
                'Prediction case malformed: ' + sid)
        for m in c['methods'].values():
            require((m['index'] is None and m['point'] is None) or
                    (isinstance(m['index'], int) and 0 <= m['index'] < e['n']
                     and len(m['point']) == 2), 'Invalid method point: ' + sid)
        cases[sid] = c
    print('Gate passed: all 300 per-case SHA receipts', flush=True)
    return p, entries, cases


def interval(delta, entries, subset):
    """Source-stratified RGB-cluster bootstrap of a query-mean paired delta."""
    indices = [i for i, e in enumerate(entries) if subset == 'all300' or e['source'] == subset]
    source_groups = defaultdict(lambda: defaultdict(list))
    for i in indices:
        e = entries[i]
        source_groups[e['source']][e['group']].append(i)
    rng = np.random.default_rng(BOOT_SEED)
    numer = np.zeros(REPLICATES, np.float64)
    denom = np.zeros(REPLICATES, np.float64)
    for source in sorted(source_groups):
        clusters = list(source_groups[source].values())
        m = len(clusters)
        counts = rng.multinomial(m, np.full(m, 1 / m), size=REPLICATES)
        cluster_num = np.asarray([delta[v].sum() for v in clusters], np.float64)
        cluster_den = np.asarray([len(v) for v in clusters], np.float64)
        numer += counts @ cluster_num
        denom += counts @ cluster_den
    samples = numer / denom
    return dict(delta=float(np.mean(delta[indices])),
                ci95=np.quantile(samples, [0.025, 0.975]).tolist(),
                ci98_333=np.quantile(samples, [0.05 / 6, 1 - 0.05 / 6]).tolist(),
                n=len(indices), rgb_clusters=sum(len(v) for v in source_groups.values()),
                bootstrap_replicates=REPLICATES, bootstrap_seed=BOOT_SEED)


def analyze():
    p, entries, cases = gate()
    manifest = [json.loads(line) for line in MANIFEST.read_text(encoding='utf-8').splitlines() if line]
    require([r['sample_id'] for r in manifest] == p['sample_ids'], 'GT manifest order changed')
    mvp_rows = read(MVP)['rows']
    gui_rows = read(GUI)['rows']
    require([r['sample_id'] for r in mvp_rows] == p['sample_ids'] and
            [r['sample_id'] for r in gui_rows] == p['sample_ids'],
            'V17 external baseline order differs from registered V16 300')
    hits = {name: [] for name in METHODS}
    hits.update(MVP=[], GUI_Actor_K16=[])
    details = []
    k16_reach = []
    mvp_reach = []
    for e, row, mvp, gui in zip(entries, manifest, mvp_rows, gui_rows):
        sid = e['sample_id']
        width, height = row['image_size']
        with np.load(e['path'], allow_pickle=False) as z:
            points, baseline = z['points'], z['baseline']
        labels = np.load(e['labels_path'], allow_pickle=False)
        require(sha(e['labels_path']) == e['labels_sha256'] and len(labels) == e['n'],
                'V16 label cache changed: ' + sid)
        def hit(point):
            return bool(point is not None and point_in_geometry(
                (point[0] / width, point[1] / height), row['target_geometry']))
        k16 = bool(labels[baseline].any()) if len(baseline) else False
        require(k16 == any(hit(points[int(ix)].tolist()) for ix in baseline),
                'K16 reachability/geometry disagreement: ' + sid)
        mp = [v['point'] for v in mvp['output']['reference_result']['all_predictions']]
        mreach = any(hit(point) for point in mp)
        require(len(mp) == 3 and mvp['hit'] == float(hit(mvp['point']))
                and gui['hit'] == float(hit(gui['point'])),
                'V17 baseline geometry/coverage mismatch: ' + sid)
        k16_reach.append(k16)
        mvp_reach.append(mreach)
        outcomes = {}
        for name in METHODS:
            prediction = cases[sid]['methods'][name]
            idx, point = prediction['index'], prediction['point']
            require((idx is None and point is None and len(baseline) == 0) or
                    (idx is not None and int(idx) in baseline and
                     point == points[int(idx)].tolist()),
                    'Not original-K16 forced Top1: ' + sid + '/' + name)
            result = hit(point)
            require(result == bool(labels[idx]) if idx is not None else not result,
                    'V16 cached label/geometry mismatch: ' + sid + '/' + name)
            hits[name].append(result)
            outcomes[name] = result
        hits['MVP'].append(hit(mvp['point']))
        hits['GUI_Actor_K16'].append(hit(gui['point']))
        outcomes['MVP'] = hits['MVP'][-1]
        outcomes['GUI_Actor_K16'] = hits['GUI_Actor_K16'][-1]
        details.append(dict(sample_id=sid, source=e['source'], rgb_group=e['group'],
                            K16_reachable=k16, MVP_three_proposal_reachable=mreach,
                            predictions=cases[sid]['methods'], hits=outcomes,
                            external_points={'MVP': mvp['point'], 'GUI_Actor_K16': gui['point']}))
    require(sum(k16_reach) == 152 and sum(mvp_reach) == 151 and
            sum(a and b for a, b in zip(k16_reach, mvp_reach)) == 105,
            'V17 fixed reachability decomposition did not reproduce')
    by_subset = {}
    for subset in ('all300', 'ScreenSpot-Pro', 'VenusBench-GD'):
        ix = [i for i, e in enumerate(entries) if subset == 'all300' or e['source'] == subset]
        methods = {name: dict(hits=int(sum(hits[name][i] for i in ix)),
                              accuracy=float(np.mean([hits[name][i] for i in ix])))
                   for name in hits}
        summaries = {}
        comparisons = {}
        for variant in VARIANTS:
            name = 'full_' + variant
            seed_hits = [methods[f'{name}_s{s}']['hits'] for s in SEEDS]
            summaries[variant] = dict(seed_hits=seed_hits,
                                      seed_mean_hits=float(np.mean(seed_hits)),
                                      seed_mean_accuracy=float(np.mean(seed_hits) / len(ix)))
        for variant in VARIANTS[1:]:
            delta = np.mean([[float(a) - float(b) for a, b in zip(
                hits[f'full_aligned_s{s}'], hits[f'full_{variant}_s{s}'])] for s in SEEDS], axis=0)
            comparisons['aligned_vs_' + variant] = interval(delta, entries, subset)
        for baseline in ('MVP', 'GUI_Actor_K16'):
            delta = np.mean([[float(a) - float(b) for a, b in zip(
                hits[f'full_aligned_s{s}'], hits[baseline])] for s in SEEDS], axis=0)
            comparisons['aligned_vs_' + baseline] = interval(delta, entries, subset)
        by_subset[subset] = dict(n=len(ix), methods=methods, variants=summaries,
                                 comparisons=comparisons)
    kix = [i for i, flag in enumerate(k16_reach) if flag]
    both = [i for i, (a, b) in enumerate(zip(k16_reach, mvp_reach)) if a and b]
    selection = {}
    for name, ix in (('K16_reachable', kix), ('both_reachable', both)):
        seed_hits = [int(sum(hits[f'full_aligned_s{s}'][i] for i in ix)) for s in SEEDS]
        selection[name] = dict(n=len(ix), aligned_seed_hits=seed_hits,
                               aligned_seed_mean_hits=float(np.mean(seed_hits)),
                               aligned_seed_mean_accuracy=float(np.mean(seed_hits) / len(ix)),
                               MVP_hits=int(sum(hits['MVP'][i] for i in ix)))
    result = dict(complete=True, scope=p['scope'], n=300, protocol_sha256=sha(PROTOCOL),
                  predictions_sha256=sha(SEAL), input_sha256=p['input_sha256'],
                  model_sha256=p['model_sha256'], by_subset=by_subset,
                  candidate_selection=selection,
                  fixed_reachability=dict(K16=sum(k16_reach), MVP=sum(mvp_reach),
                                          both=len(both)),
                  caveat='Retrospective, historically observed cohort; intervals conditional on fixed trained seeds and candidate pools.')
    write_once(OUT / 'analysis/details.json', details)
    write_once(OUT / 'analysis/results.json', result)
    lines = [
        'Full-model V16/V17 fixed 300-query retrospective supplement',
        'ScreenSpot-Pro150 + VenusBench-GD150; original K16 forced Top1; full aligned and five full ablations, S1/S2/S3.',
        'This cohort and its labels/results were historically observed. Differences and intervals are descriptive, not fresh independent confirmation.',
        f'Protocol SHA256: {sha(PROTOCOL)}', f'Prediction seal SHA256: {sha(SEAL)}',
        f'Analysis JSON: {OUT / "analysis/results.json"}',
        'Intervals: source-stratified RGB-image-cluster paired bootstrap, 10000 draws, seed 20261113; conditional on fixed three seeds.',
        'The 98.333% intervals are a three-comparison-family reference for comparison with older tables; no new confirmatory family claim.',
        '',
    ]
    for subset, data in by_subset.items():
        lines += [f'{subset}: n={data["n"]}']
        for v in VARIANTS:
            x = data['variants'][v]
            lines += [f'  full {v}: seed hits {x["seed_hits"]}; mean {x["seed_mean_hits"]:.2f}/{data["n"]} = {100*x["seed_mean_accuracy"]:.2f}%']
        for name in ('aligned_vs_no_evidence', 'aligned_vs_spatial_shuffle',
                     'aligned_vs_MVP', 'aligned_vs_GUI_Actor_K16'):
            x = data['comparisons'][name]
            lines += [f'  {name}: {100*x["delta"]:+.2f} pp; 95% [{100*x["ci95"][0]:+.2f}, {100*x["ci95"][1]:+.2f}]; 98.333% [{100*x["ci98_333"][0]:+.2f}, {100*x["ci98_333"][1]:+.2f}] pp']
        lines += ['']
    for name, x in selection.items():
        lines += [f'{name}: n={x["n"]}, full aligned seed hits {x["aligned_seed_hits"]}, mean {x["aligned_seed_mean_hits"]:.2f}; MVP hits {x["MVP_hits"]}']
    text = '\n'.join(lines) + '\n'
    path = OUT / 'analysis/readout.txt'
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        require(path.read_text(encoding='utf-8') == text, 'Readout differs from prior analysis')
    else:
        path.write_text(text, encoding='utf-8')
    print(text, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', required=True,
                    choices=('register', 'predict', 'verify-gate', 'analyze'))
    args = ap.parse_args()
    if args.stage == 'register':
        register()
    elif args.stage == 'predict':
        predict()
    elif args.stage == 'verify-gate':
        gate()
    else:
        analyze()


if __name__ == '__main__':
    main()
