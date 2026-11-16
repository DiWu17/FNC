"""Paper scale ablations: archived DEV parity, then label-free held-out inference.

This runner never changes V15/V16/V18/V32 artifacts. Reserve and full-benchmark
labels are opened only by the explicit ``analyze-*`` stages, after a complete
prediction seal exists. The main accuracy metric is original-K16 forced Top1.
"""
from __future__ import annotations

import os
os.environ.setdefault('PYTHONDONTWRITEBYTECODE', '1')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
import numpy as np
from PIL import Image
import torch

import run_quality_readout_v15_3k as prior
from clickagent.contribution_ablation import VARIANTS, transform_inputs
from clickagent.evidence_retention import RetentionModel, STATE_DIM, build_base, state_features, require
from clickagent.public_evaluation import point_in_geometry

RUN = ROOT / 'runs/paper_scale_ablation_v1'
OUT = RUN / 'eval'
MAIN = ROOT / 'runs/quality_readout_v15_3k'
V16 = ROOT / 'runs/contribution_v16'
V18 = ROOT / 'runs/full_scaling_v18'
V32 = ROOT / 'runs/independent_diagnostic_v32'
V18_CONFIRM = ROOT / 'runs/mvp_confirmation_v18'
SSPRO = ROOT / 'data/public_v1/manifests/screenspot_pro.jsonl'
IMAGE_INDEX = ROOT / 'data/conditioned_v3/public_image_index.json'
SEEDS = (20261311, 20261312, 20261313)
PUBLIC_KEYS = ('sample_id', 'dataset', 'image', 'image_size', 'image_sha256', 'instruction')
CASE_KEYS = PUBLIC_KEYS + ('pixel_sha256', 'application', 'application_group', 'ui_type')
read, write, sha, key, digest = prior.read, prior.write, prior.sha, prior.key, prior.digest


def stamp():
    return datetime.now(timezone.utc).isoformat()


def log(stage, **fields):
    row = dict(stage=stage, at_utc=stamp(), **fields)
    p = OUT / 'events.jsonl'
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('a', encoding='utf-8') as f:
        f.write(json.dumps(row, ensure_ascii=False) + '\n')
    print(json.dumps(row, ensure_ascii=False), flush=True)


def verify_sources():
    for name, expected in read(MAIN / 'protocol.json')['source_identity'].items():
        require(sha(ROOT / name) == expected, 'V15 scientific source changed: ' + name)
    for name, expected in read(V16 / 'protocol.json')['source_identity'].items():
        require(sha(ROOT / name) == expected, 'V16 scientific source changed: ' + name)
    for name, expected in read(V18 / 'protocol.json')['hashes'].items():
        # V18 seals its own script and scientific dependencies, not its huge caches.
        if name.endswith(('.py', '.pt')):
            require(sha(name) == expected, 'V18 scientific source or proposer changed: ' + name)


def models(reference_only=False):
    """One canonical model table; no model or seed chosen from outcomes."""
    a = read(V16 / 'models/frozen.json')
    b = read(V18 / 'models/frozen.json')
    require(a['complete'] and b['complete'], 'Reference model seals incomplete')
    result = {}
    for variant in VARIANTS:
        for s, seed in enumerate(SEEDS, 1):
            source = a['models'][f'{variant}_s{s}']
            require(source['variant'] == variant and source['seed'] == seed, '3k model identity mismatch')
            result[f'3k_{variant}_s{s}'] = dict(source, scale='3k', steps=2976, queries=2973)
    for s, seed in enumerate(SEEDS, 1):
        source = b['models'][f'full_s{s}']
        require(source['steps'] == 13376 and source['queries'] == 13373,
                'V18 full model training contract mismatch')
        result[f'full_aligned_s{s}'] = dict(source, variant='aligned', seed=seed, scale='full')
        control = b['models'][f'matched3k_s{s}']
        require(control['steps'] == 13376 and control['queries'] == 2973,
                'V18 equal-update 3k control contract mismatch')
        result[f'matched3k_aligned_s{s}'] = dict(control, variant='aligned', seed=seed,
                                                  scale='matched3k')
    if not reference_only:
        p = RUN / 'models/frozen.json'
        require(p.exists(), 'Full ablation seal missing: ' + str(p))
        c = read(p)
        require(c['complete'], 'Full ablation seal incomplete')
        require(c['protocol_sha256'] == sha(RUN / 'protocol.json'),
                'Full ablation model/protocol seal differs')
        for source_name, expected_sha in c['source_identity'].items():
            require(sha(ROOT / source_name) == expected_sha,
                    'Full ablation training source changed: ' + source_name)
        expected = {f'full_{v}_s{s}' for v in VARIANTS for s in (1, 2, 3)}
        require(set(c['models']) == expected, 'Full ablation seal must have all 18 fixed slots')
        for variant in VARIANTS:
            for s, seed in enumerate(SEEDS, 1):
                name = f'full_{variant}_s{s}'
                source = c['models'][name]
                require(source['variant'] == variant and source['seed'] == seed,
                        'Full ablation model identity mismatch: ' + name)
                require(source['steps'] == 13376 and source['queries'] == 13373,
                        'Full ablation training budget mismatch: ' + name)
                if variant == 'aligned':
                    require(source['sha256'] == result[name]['sha256'],
                            'Full aligned reference differs from V18: ' + name)
                else:
                    result[name] = dict(source, scale='full')
    for name, entry in result.items():
        require(sha(entry['path']) == entry['sha256'], 'Checkpoint changed: ' + name)
    return result


@contextmanager
def extraction_output(path):
    old = prior.OUT
    prior.OUT = path
    try:
        yield
    finally:
        prior.OUT = old


def public_case(row):
    require(all(k in row for k in PUBLIC_KEYS), 'Missing public field')
    return {k: row[k] for k in PUBLIC_KEYS}


def compact_public(pipe, image_size):
    """V15 feature construction without geometry or a labels array."""
    base = build_base(pipe, image_size)
    views = [dict(feature_path=pipe['global_feature_path'],
                  feature_sha256=pipe['global_feature_sha256'], transform=None)] + pipe['local_views']
    features, masks = [], []
    for view in views:
        require(sha(view['feature_path']) == view['feature_sha256'], 'Raw view changed')
        with np.load(view['feature_path'], allow_pickle=False) as z:
            meta = json.loads(str(z['metadata']))
            arrays = {k: z[k].copy() for k in ('target_attention', 'target_hidden', 'image_hidden')}
        f, m = state_features(arrays, meta, base['points'], image_size, view['transform'])
        features.append(f)
        masks.append(m)
    tmax = max(1, *(f.shape[1] for f in features))
    state = np.zeros((len(base['points']), 4, tmax, STATE_DIM), np.float16)
    mask = np.zeros((len(base['points']), 4, tmax), bool)
    for j, (f, m) in enumerate(zip(features, masks)):
        t = f.shape[1]
        state[:, j, :t] = f
        mask[:, j, :t] = m[:, None]
    return base, state, mask


def archived_sources():
    """Previously observed cases only; held-out reserve/venus may never match."""
    v16 = {e['sample_id']: e for e in read(V16 / 'independent/index.json')['cases']}
    v18 = {r['sample_id']: V18_CONFIRM / f'features/pipeline/{key(r["sample_id"])}.json'
           for r in read(V18_CONFIRM / 'predictions/ours.json')['rows']}
    return v16, v18


def authenticate_pipeline(path, public):
    pipe = read(path)
    require(pipe['complete'] and pipe['input_identity'] == digest(public) and
            pipe['ground_truth_in_model_inputs'] is False,
            'Archived pipeline/public identity mismatch: ' + str(path))
    require(pipe['proposer_sha256'] == sha(prior.base.V12 / 'proposers/full.pt'),
            'Archived proposal checkpoint changed')
    views = [dict(feature_path=pipe['global_feature_path'],
                  feature_sha256=pipe['global_feature_sha256'])] + pipe['local_views']
    for view in views:
        require(sha(view['feature_path']) == view['feature_sha256'],
                'Archived Qwen raw view changed')
    return pipe


def feature_from_v16(entry, public):
    """Refer only to V16's public base/state/mask; never load its label array."""
    pipe_path = Path(entry['pipeline_path'])
    authenticate_pipeline(pipe_path, public)
    for path_field, hash_field in (('path', 'sha256'), ('state_path', 'state_sha256'),
                                   ('mask_path', 'mask_sha256'),
                                   ('pipeline_path', 'pipeline_sha256')):
        require(sha(entry[path_field]) == entry[hash_field],
                'Archived public feature changed: ' + entry[path_field])
    return dict(sample_id=public['sample_id'], case_identity=digest(public),
                n=entry['n'], path=entry['path'], sha256=entry['sha256'],
                state_path=entry['state_path'], state_sha256=entry['state_sha256'],
                mask_path=entry['mask_path'], mask_sha256=entry['mask_sha256'],
                pipeline_path=entry['pipeline_path'], pipeline_sha256=entry['pipeline_sha256'],
                labels_present=False, ground_truth_in_model_input=False,
                public_feature_reuse='V16 independent, previously observed')


def save_array(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    with temp.open('wb') as f:
        np.save(f, value, allow_pickle=False)
    prior.base.replace(temp, path)


def authenticate_feature(entry, row):
    require(entry['sample_id'] == row['sample_id'] and
            entry['case_identity'] == digest(public_case(row)), 'Feature case identity mismatch')
    require(entry['labels_present'] is False, 'Held-out feature includes labels')
    for path_field, hash_field in (('path', 'sha256'), ('state_path', 'state_sha256'),
                                   ('mask_path', 'mask_sha256'), ('pipeline_path', 'pipeline_sha256')):
        require(sha(entry[path_field]) == entry[hash_field], 'Feature changed: ' + entry[path_field])


def read_feature(entry, row):
    authenticate_feature(entry, row)
    with np.load(entry['path'], allow_pickle=False) as z:
        r = {k: z[k].copy() for k in ('points', 'base', 'active', 'baseline')}
    r.update(entry)
    r['state'] = np.load(entry['state_path'], mmap_mode='r', allow_pickle=False)
    r['mask'] = np.load(entry['mask_path'], mmap_mode='r', allow_pickle=False)
    require(r['base'].shape == (entry['n'], 32) and
            r['state'].shape[0] == entry['n'] and
            r['mask'].shape == r['state'].shape[:-1], 'Feature shape mismatch')
    require((r['baseline'] < entry['n']).all() and (r['baseline'] >= 0).all(),
            'K16 pool index outside candidate set')
    return r


def load_model_table(table, device):
    result = {}
    for name, entry in table.items():
        saved = torch.load(entry['path'], map_location='cpu', weights_only=True)
        model = RetentionModel().to(device).eval()
        model.load_state_dict(saved['state_dict'])
        require(sum(p.numel() for p in model.parameters()) == 22050, 'Readout size changed')
        result[name] = (entry, model, saved['mean'].numpy(), saved['std'].numpy())
    return result


@torch.inference_mode()
def score_record(record, model_table, device):
    """For each model, use its own training-only scaler and original K16 tie order."""
    output = {}
    for name, (entry, model, mean, std) in model_table.items():
        if record['n']:
            args, lengths = prior.tensors([record], mean, std, 'aligned', device)
            args = transform_inputs(args, lengths, [record['sample_id']], entry['variant'])
            score = model(**args).float().cpu().numpy()
            require(score.shape == (record['n'],) and np.isfinite(score).all(),
                    'Invalid score vector: ' + name)
            pool = np.sort(record['baseline'])
            pick = int(pool[np.argsort(-score[pool], kind='stable')[0]]) if len(pool) else None
        else:
            pick = None
        point = record['points'][pick].tolist() if pick is not None else None
        output[name] = dict(index=pick, point=point)
    return output


def dev_entries(cohort):
    require(cohort in ('old_val', 'new_development'), 'Unknown development cohort')
    return (prior.base.old_entries('val') if cohort == 'old_val' else
            read(MAIN / 'features/new_reserve/index.json')['cases'])


def parity(cohort, device, limit, reference_only):
    verify_sources()
    table = models(reference_only)
    entries = dev_entries(cohort)
    if limit:
        require(limit > 0, 'Limit must be positive')
        entries = entries[:limit]
    folder = ('parity_reference' if reference_only else 'parity_full')
    result_path = OUT / (folder if not limit else folder + '_smoke') / f'{cohort}.json'
    if result_path.exists() and not limit:
        old = read(result_path)
        require(old['complete'] and old['script_sha256'] == sha(__file__) and
                old['model_hashes'] == {n: e['sha256'] for n, e in table.items()},
                'Existing parity seal mismatch')
        return
    records = prior.base.load_records(entries)
    loaded = load_model_table(table, device)
    v16 = {r['sample_id']: r for r in read(V16 / f'evaluation/{cohort}/details.json')}
    v18 = read(V18 / f'evaluation/{cohort}.json')
    v18_order = {sid: i for i, sid in enumerate(v18['sample_ids'])}
    expected_3k, expected_full = 0, 0
    all_rows = []
    for i, r in enumerate(records, 1):
        scored = score_record(r, loaded, device)
        checks = {}
        for variant in VARIANTS:
            for s in (1, 2, 3):
                name = f'3k_{variant}_s{s}'
                historical = v16[r['sample_id']]['methods'][f'{variant}_s{s}']
                observed = scored[name]['point']
                same_point = observed == historical['point']
                hit = bool(scored[name]['index'] is not None and r['labels'][scored[name]['index']])
                checks[name] = same_point and hit == historical['top1']
                expected_3k += 1
        for s in (1, 2, 3):
            ix = v18_order[r['sample_id']]
            for name, archive in ((f'full_aligned_s{s}', f'full_s{s}'),
                                  (f'matched3k_aligned_s{s}', f'matched3k_s{s}')):
                historical_hit = v18['K16_hit_vectors'][archive][ix]
                hit = bool(scored[name]['index'] is not None and r['labels'][scored[name]['index']])
                checks[name] = hit == historical_hit
                expected_full += 1
        require(all(checks.values()), f'Archived parity mismatch: {r["sample_id"]}, '
                f'{[n for n, ok in checks.items() if not ok]}')
        all_rows.append(dict(sample_id=r['sample_id'], checks=checks, predictions=scored))
        if i % 25 == 0 or i == len(records):
            log('parity', cohort=cohort, completed=i, total=len(records), models=len(table))
    write(result_path, dict(complete=not bool(limit), smoke=bool(limit), cohort=cohort,
                            script_sha256=sha(__file__),
                            n=len(records), expected_3k_checks=expected_3k,
                            expected_full_checks=expected_full,
                            model_hashes={n: e['sha256'] for n, e in table.items()},
                            archived_prediction_agreement=True, rows=all_rows))


def feature_parity(per_source):
    """Rebuild public-only tensors from archived DEV raw views, without a VLM call."""
    verify_sources()
    require(per_source >= 1, 'Need at least one existing DEV case per source')
    entries = read(MAIN / 'features/new_reserve/index.json')['cases']
    by_source = {}
    for entry in entries:
        by_source.setdefault(entry['source'], []).append(entry)
    require(set(by_source) == {'GroundCUA', 'ShowUI-web', 'AMEX'},
            'Unexpected DEV source inventory')
    chosen = [entry for source in sorted(by_source)
              for entry in by_source[source][:per_source]]
    result = []
    for entry in chosen:
        require(sha(entry['pipeline_path']) == entry['pipeline_sha256'],
                'Archived DEV pipeline changed')
        pipe = read(entry['pipeline_path'])
        with np.load(entry['path'], allow_pickle=False) as z:
            old = {k: z[k].copy() for k in ('points', 'base', 'active', 'baseline')}
        image_size = [int(x) for x in pipe['global_result']['image_size']] if (
            'image_size' in pipe['global_result']) else None
        if image_size is None:
            # The public-only pipeline does not persist dimensions in the
            # proposal payload; obtain them from the already observed DEV row.
            dev = {q['sample_id']: q for q in prior.base.rows(MAIN / 'data/new_reserve.jsonl')}
            image_size = dev[entry['sample_id']]['image_size']
        base, state, mask = compact_public(pipe, image_size)
        for name, value in base.items():
            require(np.array_equal(value, old[name]),
                    'Public feature parity failed for ' + entry['sample_id'] + ': ' + name)
        require(np.array_equal(state, np.load(entry['state_path'], allow_pickle=False)),
                'Public state parity failed for ' + entry['sample_id'])
        require(np.array_equal(mask, np.load(entry['mask_path'], allow_pickle=False)),
                'Public mask parity failed for ' + entry['sample_id'])
        result.append(dict(sample_id=entry['sample_id'], source=entry['source'],
                           candidates=len(base['points']), state_shape=list(state.shape)))
        log('feature_parity', completed=len(result), total=len(chosen))
    write(OUT / 'parity_features.json', dict(complete=True, n=len(result),
                                             script_sha256=sha(__file__),
                                             sources=sorted(by_source), per_source=per_source,
                                             exact_base_state_mask=True, rows=result))


def archive_parity():
    """Validate identity and exact compact-feature reuse on observed SSP cases."""
    v16, v18 = archived_sources()
    public_rows = {q['sample_id']: q for q in read(RUN / 'data/full_public.json')}
    a = sorted(s for s in v16 if s.startswith('screenspot_pro:'))
    b = sorted(s for s in v18 if s.startswith('screenspot_pro:'))
    require(len(a) == 150 and len(b) == 140 and not set(a) & set(b),
            'Archived SSP reuse inventory changed')
    checks = []
    for sid in a[:3]:
        row = public_rows[sid]
        pub = public_case(row)
        e = feature_from_v16(v16[sid], pub)
        pipe = authenticate_pipeline(Path(e['pipeline_path']), pub)
        base, state, mask = compact_public(pipe, row['image_size'])
        with np.load(e['path'], allow_pickle=False) as z:
            for name, array in base.items():
                require(np.array_equal(array, z[name]), 'V16 reuse base mismatch: ' + sid)
        require(np.array_equal(state, np.load(e['state_path'], allow_pickle=False)) and
                np.array_equal(mask, np.load(e['mask_path'], allow_pickle=False)),
                'V16 reuse state/mask mismatch: ' + sid)
        checks.append(dict(sample_id=sid, origin='V16', candidates=len(base['points'])))
    for sid in b[:3]:
        row = public_rows[sid]
        pipe = authenticate_pipeline(v18[sid], public_case(row))
        base, state, mask = compact_public(pipe, row['image_size'])
        require(state.shape[:-1] == mask.shape and len(base['points']) == len(state),
                'V18 pipeline public compact invalid: ' + sid)
        checks.append(dict(sample_id=sid, origin='V18', candidates=len(base['points'])))
    write(OUT / 'parity_archives.json', dict(complete=True, v16_reusable=150,
                                            v18_pipeline_reusable=140,
                                            script_sha256=sha(__file__),
                                            overlap=0, checked=checks,
                                            labels_opened=False, model_forward_calls=0))
    log('archive_parity', v16=150, v18=140, checked=len(checks))


def all_models_sealed():
    verify_sources()
    table = models(False)
    require(len(table) == 39, 'Both scales must have six variants and three seeds, plus matched3k control')
    fp = OUT / 'parity_features.json'
    require(fp.exists() and read(fp)['complete'] and read(fp)['exact_base_state_mask'] and
            read(fp)['per_source'] >= 3 and read(fp)['n'] >= 9 and
            read(fp)['script_sha256'] == sha(__file__),
            'Run public-only DEV feature parity before held-out extraction')
    ap = OUT / 'parity_archives.json'
    require(ap.exists() and read(ap)['complete'] and
            read(ap)['v16_reusable'] == 150 and read(ap)['v18_pipeline_reusable'] == 140 and
            read(ap)['script_sha256'] == sha(__file__),
            'Run archived SSP public feature parity before held-out extraction')
    expected = {n: e['sha256'] for n, e in table.items()}
    for cohort in ('old_val', 'new_development'):
        p = OUT / 'parity_full' / f'{cohort}.json'
        require(p.exists(), 'Run full DEV parity first: ' + cohort)
        row = read(p)
        require(row['complete'] and row['model_hashes'] == expected and
                row['archived_prediction_agreement'] and
                row['script_sha256'] == sha(__file__),
                'DEV parity identity mismatch: ' + cohort)
    return table


def prepare_public(cohort):
    table = all_models_sealed()
    folder = OUT / cohort
    protocol_path = folder / 'protocol.json'
    if protocol_path.exists():
        protocol = read(protocol_path)
        require(protocol['complete'] and protocol['model_hashes'] ==
                {n: e['sha256'] for n, e in table.items()}, 'Public protocol drift')
        require(sha(folder / 'public.json') == protocol['public_sha256'], 'Public cohort changed')
        return
    audit_path = RUN / 'data/audit.json'
    require(audit_path.exists(), 'Complete dataset and history audit before held-out inference')
    audit = read(audit_path)
    require(audit['training']['reserve_no_exact_or_near_training_overlap'] and
            audit['integrity']['all_official_image_file_sha256_verified'] and
            audit['integrity']['reserve_public_matches_official_index'] and
            audit['history']['pre_v32_unresolved'] == 0,
            'Dataset audit did not pass')
    if cohort == 'reserve':
        source = RUN / 'data/reserve_public.json'
        selected = read(source)
        require(len(selected) == 58, 'V32 reserve size changed')
        require(all('target_geometry' not in q and 'labels' not in q for q in selected),
                'Reserve public data contains labels')
        rows = [{k: q[k] for k in CASE_KEYS} for q in selected]
        original_inputs = {str(source): sha(source), str(V32 / 'data/reserve_public.json'):
                           sha(V32 / 'data/reserve_public.json'), str(audit_path): sha(audit_path)}
    elif cohort == 'venus':
        require((OUT / 'reserve/analysis.json').exists(),
                'Seal and analyze the 58-image reserve before cross-source confirmation')
        venus_audit = RUN / 'data/venus_audit.json'
        source = RUN / 'data/venus_public.json'
        require(source.exists() and venus_audit.exists(), 'Venus public audit not ready')
        venus_spec = read(venus_audit)
        require(venus_spec.get('complete') is True and
                venus_spec.get('independent_history_audit_passed') is True,
                'Venus independent-history audit incomplete')
        selected = read(source)
        require(len(selected) == 143, 'Venus confirmation cohort changed')
        require(all('target_geometry' not in q and 'labels' not in q for q in selected),
                'Venus public data contains labels')
        rows = [{k: q[k] for k in CASE_KEYS} for q in selected]
        require(sha(source) == venus_spec['public_sha256'], 'Venus public cohort changed')
        original_inputs = {str(source): sha(source), str(venus_audit): sha(venus_audit)}
    elif cohort == 'full':
        require((OUT / 'reserve/analysis.json').exists() and
                (OUT / 'venus/analysis.json').exists(),
                'Finish both independent confirmations before the overlapping full benchmark')
        source = RUN / 'data/full_public.json'
        selected = read(source)
        require(all('target_geometry' not in q and 'labels' not in q for q in selected),
                'Full public data contains labels')
        rows = [{k: q[k] for k in CASE_KEYS} for q in selected]
        require(len(rows) == 1581, 'ScreenSpot-Pro official count changed')
        # The official JSONL contains GT. Carry forward the preparation audit's
        # sealed digest; do not open that file until all predictions are sealed.
        official_sha = audit['source_sha256']['data/public_v1/manifests/screenspot_pro.jsonl']
        original_inputs = {str(source): sha(source), str(SSPRO): official_sha,
                           str(IMAGE_INDEX): sha(IMAGE_INDEX), str(audit_path): sha(audit_path)}
    else:
        raise ValueError('Unknown cohort: ' + cohort)
    require(len({q['sample_id'] for q in rows}) == len(rows), 'Repeated sample ID')
    require(all(set(q) == set(CASE_KEYS) for q in rows), 'Public whitelist mismatch')
    write(folder / 'public.json', rows)
    spec = dict(complete=True, cohort=cohort, n=len(rows),
                public_sha256=sha(folder / 'public.json'), source_sha256=original_inputs,
                model_hashes={n: e['sha256'] for n, e in table.items()},
                model_seal_sha256=sha(RUN / 'models/frozen.json'),
                v16_seal_sha256=sha(V16 / 'models/frozen.json'),
                v18_seal_sha256=sha(V18 / 'models/frozen.json'),
                script_sha256=sha(__file__), metric='Original K16 forced Top1, canonical union-index tie',
                public_only=True, labels_opened=False, created_at_utc=stamp())
    write(protocol_path, spec)
    log('prepared_public', cohort=cohort, n=len(rows))


def verify_public(cohort):
    folder = OUT / cohort
    spec = read(folder / 'protocol.json')
    require(spec['complete'] and spec['cohort'] == cohort and spec['script_sha256'] == sha(__file__),
            'Public protocol/source mismatch')
    for path, expected in spec['source_sha256'].items():
        if cohort == 'full' and path == str(SSPRO):
            # Full benchmark labels are authenticated and opened only in analyze().
            continue
        require(sha(path) == expected, 'Public source changed: ' + path)
    require(sha(folder / 'public.json') == spec['public_sha256'], 'Public cohort changed')
    require(spec['model_hashes'] == {n: e['sha256'] for n, e in models(False).items()},
            'Model seal changed since public registration')
    require(spec['model_seal_sha256'] == sha(RUN / 'models/frozen.json') and
            spec['v16_seal_sha256'] == sha(V16 / 'models/frozen.json') and
            spec['v18_seal_sha256'] == sha(V18 / 'models/frozen.json'),
            'Model registry seal changed since public registration')
    rows = read(folder / 'public.json')
    require(len(rows) == spec['n'] and all(set(q) == set(CASE_KEYS) for q in rows),
            'Public cohort is not label-free')
    return spec, rows


def extract_public(cohort):
    spec, rows = verify_public(cohort)
    folder = OUT / cohort
    index_path = folder / 'features/index.json'
    if index_path.exists():
        old = read(index_path)
        require(old['complete'] and old['public_sha256'] == spec['public_sha256'],
                'Existing feature index mismatch')
        return
    entries = []
    extractor = None
    v16, v18 = archived_sources()
    if cohort in ('reserve', 'venus'):
        require(not ({q['sample_id'] for q in rows} & (set(v16) | set(v18))),
                'Independent cohort unexpectedly intersects archived feature IDs')
    reused_v16 = reused_v18 = fresh = 0
    started = time.perf_counter()
    with extraction_output(folder):
        for i, row in enumerate(rows, 1):
            p = folder / f'features/cases/{key(row["sample_id"])}.json'
            if p.exists():
                entry = read(p)
                authenticate_feature(entry, row)
            else:
                pub = public_case(row)
                if row['sample_id'] in v16:
                    entry = feature_from_v16(v16[row['sample_id']], pub)
                    reused_v16 += 1
                else:
                    if row['sample_id'] in v18:
                        pipe_path = v18[row['sample_id']]
                        pipe = authenticate_pipeline(pipe_path, pub)
                        reused_v18 += 1
                    else:
                        if extractor is None:
                            extractor = prior.Extractor()
                        pipe, pipe_path = extractor.case(pub)
                        fresh += 1
                    base, state, mask = compact_public(pipe, pub['image_size'])
                    bp = folder / f'features/arrays/{key(row["sample_id"])}.npz'
                    prior.base.npz(bp, **base)
                    sp = bp.with_suffix('.state.npy')
                    mp = bp.with_suffix('.mask.npy')
                    save_array(sp, state)
                    save_array(mp, mask)
                    entry = dict(sample_id=pub['sample_id'], case_identity=digest(pub),
                                 n=len(base['points']), path=str(bp), sha256=sha(bp),
                                 state_path=str(sp), state_sha256=sha(sp),
                                 mask_path=str(mp), mask_sha256=sha(mp),
                                 pipeline_path=str(pipe_path), pipeline_sha256=sha(pipe_path),
                                 labels_present=False, ground_truth_in_model_input=False,
                                 public_feature_reuse=('V18 original pipeline, previously observed'
                                                       if row['sample_id'] in v18 else None))
                write(p, entry)
            entries.append(entry)
            if i % 5 == 0 or i == len(rows):
                log('extract_public', cohort=cohort, completed=i, total=len(rows),
                    reused_v16=reused_v16, reused_v18=reused_v18, fresh=fresh,
                    elapsed_seconds=round(time.perf_counter() - started, 1))
    del extractor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    write(index_path, dict(complete=True, cohort=cohort, n=len(entries), cases=entries,
                           public_sha256=spec['public_sha256'], labels_present=False,
                           ground_truth_in_model_input=False,
                           reused_v16=reused_v16, reused_v18=reused_v18, fresh=fresh))


def score_public(cohort, device):
    spec, rows = verify_public(cohort)
    folder = OUT / cohort
    index = read(folder / 'features/index.json')
    require(index['complete'] and index['n'] == len(rows) and
            index['public_sha256'] == spec['public_sha256'] and
            index['labels_present'] is False, 'Label-free feature index not complete')
    entries = index['cases']
    require([q['sample_id'] for q in entries] == [q['sample_id'] for q in rows],
            'Feature/public order mismatch')
    output = folder / 'predictions.json'
    if output.exists():
        old = read(output)
        require(old['complete'] and old['public_sha256'] == spec['public_sha256'] and
                old['model_hashes'] == spec['model_hashes'], 'Prediction seal mismatch')
        for row in rows:
            sid = row['sample_id']
            p = folder / f'predictions/cases/{key(sid)}.json'
            require(sha(p) == old['case_hashes'][sid],
                    'Sealed prediction case changed: ' + sid)
        return
    loaded = load_model_table(models(False), device)
    case_hashes = {}
    started = time.perf_counter()
    for i, (row, entry) in enumerate(zip(rows, entries), 1):
        p = folder / f'predictions/cases/{key(row["sample_id"])}.json'
        if p.exists():
            saved = read(p)
            require(saved['sample_id'] == row['sample_id'] and
                    saved['feature_sha256'] == entry['sha256'] and
                    saved['model_hashes'] == spec['model_hashes'] and
                    saved['complete'] and saved['labels_opened'] is False,
                    'Cached score case mismatch')
        else:
            r = read_feature(entry, row)
            scored = score_record(r, loaded, device)
            saved = dict(complete=True, sample_id=row['sample_id'],
                         public_identity=digest(public_case(row)),
                         feature_sha256=entry['sha256'], model_hashes=spec['model_hashes'],
                         candidate_count=r['n'], K16_count=len(r['baseline']),
                         methods=scored, labels_opened=False)
            write(p, saved)
        case_hashes[row['sample_id']] = sha(p)
        if i % 25 == 0 or i == len(rows):
            log('score_public', cohort=cohort, completed=i, total=len(rows),
                elapsed_seconds=round(time.perf_counter() - started, 1))
    write(output, dict(complete=True, cohort=cohort, n=len(rows),
                       public_sha256=spec['public_sha256'],
                       model_hashes=spec['model_hashes'], case_hashes=case_hashes,
                       labels_opened=False, scored_at_utc=stamp()))
    log('predictions_sealed', cohort=cohort, n=len(rows), sha256=sha(output))


def paired_intervals(hits, rows):
    """Paired inference; an RGB image is one bootstrap unit across its queries."""
    groups = {}
    for i, row in enumerate(rows):
        groups.setdefault(row['pixel_sha256'], []).append(i)
    clusters = list(groups.values())
    n = len(clusters)
    rng = np.random.default_rng(20261111)
    draws = rng.multinomial(n, np.full(n, 1 / n), size=10000).astype(np.float32)
    cluster_size = np.asarray([len(v) for v in clusters], np.float32)
    denominators = draws @ cluster_size

    def compare(a, b):
        a_seed = np.asarray([hits[f'{a}_s{s}'] for s in (1, 2, 3)], np.float32)
        b_seed = np.asarray([hits[f'{b}_s{s}'] for s in (1, 2, 3)], np.float32)
        per_seed = (a_seed - b_seed).mean(1).tolist()
        diff = (a_seed - b_seed).mean(0)
        numerators = np.asarray([diff[ix].sum() for ix in clusters], np.float32)
        sampled = (draws @ numerators) / denominators
        return dict(delta=float(diff.mean()), per_seed_delta=per_seed,
                    ci95=np.quantile(sampled, [.025, .975]).tolist(),
                    ci98_75=np.quantile(sampled, [.00625, .99375]).tolist(),
                    bootstrap_unit='RGB image', bootstrap_replicates=10000,
                    conditional_on_fixed_three_seeds=True)

    result = {}
    for variant in VARIANTS:
        result[f'full_vs_3k/{variant}'] = compare(f'full_{variant}', f'3k_{variant}')
    for scale in ('3k', 'full'):
        for variant in VARIANTS[1:]:
            result[f'{scale}_aligned_vs_{variant}'] = compare(
                f'{scale}_aligned', f'{scale}_{variant}')
    result['full_aligned_vs_matched3k'] = compare('full_aligned', 'matched3k_aligned')
    result['3k_aligned_vs_matched3k'] = compare('3k_aligned', 'matched3k_aligned')
    return result, n


def history_strata(details, hits):
    """Official 1581 benchmark audit: exposure strata remain descriptive."""
    path = RUN / 'data/full_history_overlap.json'
    audit = {r['sample_id']: r for r in read(path)}
    require(set(audit) == {r['sample_id'] for r in details}, 'Full history audit IDs differ')
    result = {}
    for field in ('history_status', 'trained_3k_exposure', 'trained_full_exposure'):
        result[field] = {}
        for value in sorted({r[field] for r in audit.values()}):
            ix = [i for i, row in enumerate(details)
                  if audit[row['sample_id']][field] == value]
            aligned = {}
            for scale in ('3k', 'full'):
                seed_hits = [int(sum(hits[f'{scale}_aligned_s{s}'][i] for i in ix))
                             for s in (1, 2, 3)]
                aligned[scale] = dict(seed_hits=seed_hits,
                                      seed_mean_accuracy=float(np.mean(seed_hits) / len(ix)))
            result[field][value] = dict(n=len(ix), aligned=aligned,
                                        full_minus_3k_pp=100 * (
                                            aligned['full']['seed_mean_accuracy'] -
                                            aligned['3k']['seed_mean_accuracy']))
    return result


def analyze(cohort):
    spec, rows = verify_public(cohort)
    folder = OUT / cohort
    seal_path = folder / 'predictions.json'
    seal = read(seal_path)
    require(seal['complete'] and seal['n'] == len(rows) and
            seal['public_sha256'] == spec['public_sha256'] and
            seal['model_hashes'] == spec['model_hashes'] and
            seal['labels_opened'] is False, 'Cannot unlock labels before sealed predictions')
    if cohort == 'reserve':
        labels_path = V32 / 'data/reserve_labels.json'
        expected = read(V32 / 'protocol.json')['hashes'][str(labels_path)]
        require(sha(labels_path) == expected, 'Reserve labels changed')
        labels = read(labels_path)
    elif cohort == 'venus':
        labels_path = RUN / 'data/venus_labels.json'
        audit = read(RUN / 'data/venus_audit.json')
        if 'labels_sha256' in audit:
            require(sha(labels_path) == audit['labels_sha256'], 'Venus labels changed')
        labels = read(labels_path)
    else:
        labels_path = SSPRO
        require(sha(labels_path) == spec['source_sha256'][str(labels_path)],
                'Official benchmark labels changed')
        labels = {q['sample_id']: q['target_geometry'] for q in prior.base.rows(labels_path)}
    require(set(labels) >= {q['sample_id'] for q in rows}, 'Labels missing for public cohort')
    analysis_path = folder / 'analysis.json'
    if analysis_path.exists():
        old = read(analysis_path)
        require(old['complete'] and old['predictions_sha256'] == sha(seal_path) and
                old['labels_sha256'] == sha(labels_path),
                'Existing analysis seal or labels changed')
        return
    details = []
    hits = {name: [] for name in spec['model_hashes']}
    for row in rows:
        sid = row['sample_id']
        p = folder / f'predictions/cases/{key(sid)}.json'
        require(sha(p) == seal['case_hashes'][sid], 'Sealed prediction changed')
        prediction = read(p)
        require(prediction['sample_id'] == sid and
                set(prediction['methods']) == set(hits), 'Incomplete prediction methods')
        outcomes = {}
        for name, output in prediction['methods'].items():
            point = output['point']
            value = bool(point is not None and point_in_geometry(
                (np.asarray(point) / np.asarray(row['image_size'])).tolist(), labels[sid]))
            outcomes[name] = value
            hits[name].append(value)
        details.append(dict(sample_id=sid, dataset=row['dataset'],
                            application=row['application'], application_group=row['application_group'],
                            ui_type=row['ui_type'], pixel_sha256=row['pixel_sha256'], hits=outcomes))
    methods = {name: dict(hits=sum(values), accuracy=sum(values) / len(rows))
               for name, values in hits.items()}
    summary = {}
    for scale in ('3k', 'full'):
        summary[scale] = {}
        for variant in VARIANTS:
            values = [methods[f'{scale}_{variant}_s{s}']['accuracy'] for s in (1, 2, 3)]
            summary[scale][variant] = dict(seed_accuracies=values,
                                           mean_accuracy=float(np.mean(values)))
    matched_values = [methods[f'matched3k_aligned_s{s}']['accuracy'] for s in (1, 2, 3)]
    summary['matched3k_aligned'] = dict(seed_accuracies=matched_values,
                                        mean_accuracy=float(np.mean(matched_values)))
    contrasts, rgb_images = paired_intervals(hits, rows)
    analysis = dict(complete=True, cohort=cohort, n=len(rows), metric=spec['metric'],
                    predictions_sha256=sha(seal_path), labels_sha256=sha(labels_path),
                    methods=methods, summary=summary, contrasts=contrasts,
                    unique_rgb_images=rgb_images,
                    primary_family='Four evidence contrasts: (3k/full) x (no_evidence/spatial_shuffle); '
                                   'use ci98_75 for simultaneous two-sided family coverage. '
                                   'Full-vs-3k scaling and other variants descriptive.',
                    history_strata=history_strata(details, hits) if cohort == 'full' else None,
                    scope=('V32 history-excluded prospective reserve; only 58 images' if cohort == 'reserve'
                           else 'History-excluded VenusBench-GD cross-source confirmation; 143 images'
                           if cohort == 'venus' else
                           'Full official ScreenSpot-Pro; overlaps historical observations and reserve'),
                    analyzed_at_utc=stamp())
    write(folder / 'details.json', details)
    write(analysis_path, analysis)
    log('analyzed', cohort=cohort, n=len(rows), result=str(analysis_path))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', required=True, choices=(
        'parity', 'feature-parity', 'archive-parity',
        'prepare-reserve', 'extract-reserve', 'score-reserve', 'analyze-reserve',
        'prepare-venus', 'extract-venus', 'score-venus', 'analyze-venus',
        'prepare-full', 'extract-full', 'score-full', 'analyze-full'))
    p.add_argument('--cohort', choices=('old_val', 'new_development'), default='new_development')
    p.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--limit', type=int, default=0,
                   help='DEV parity smoke only; never limits held-out cohorts')
    p.add_argument('--reference-only', action='store_true',
                   help='DEV parity against archived V16/V18 checkpoints before new full ablations finish')
    args = p.parse_args()
    torch.set_num_threads(4)
    if args.stage not in ('parity', 'feature-parity', 'archive-parity'):
        require(not args.limit and not args.reference_only,
                'Limit/reference-only are only for development parity')
    if args.stage == 'parity':
        parity(args.cohort, args.device, args.limit, args.reference_only)
    elif args.stage == 'feature-parity':
        feature_parity(args.limit or 3)
    elif args.stage == 'archive-parity':
        archive_parity()
    else:
        verb, cohort = args.stage.split('-', 1)
        if verb == 'prepare':
            prepare_public(cohort)
        elif verb == 'extract':
            extract_public(cohort)
        elif verb == 'score':
            score_public(cohort, args.device)
        else:
            analyze(cohort)


if __name__ == '__main__':
    main()
