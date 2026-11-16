"""CPU-only V12 image-group folds and strictly history-excluded fresh test.

Only this experiment's new output directory is written. Historical manifests,
raw Qwen features, checkpoints and predictions remain read-only.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from prepare_semantic_head_data import (HammingIndex, discover_history, image_info,
    key, read_rows, sha, valid_public_geometry, write_json, write_rows)
from prepare_candidate_scorer import resolve_history, path_key

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = Path('runs/crossfit_candidates_v12')
BASE = Path('runs/candidate_scorer_v11/manifest.jsonl')
SEED = 20261201
RADIUS = 4
N_FOLDS = 5
DATASETS = ('screenspot_v2', 'screenspot_pro', 'venusbench_gd')
ADDED_FIELDS = frozenset(('image_fold', 'image_fold_group', 'provider',
                         'oof_provider', 'in_sample_provider', 'raw_path'))


def digest_json(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(',', ':')).encode()).hexdigest()


def image_components(rows, radius=RADIUS):
    """RGB equality plus transitive closure of the fixed 256-bit dHash rule."""
    by_pixel = defaultdict(list)
    for row in rows:
        by_pixel[row['pixel_sha256']].append(row)
    pixels = sorted(by_pixel)
    parent = {p: p for p in pixels}
    hashes = {}
    for p in pixels:
        values = {r['dhash256'] for r in by_pixel[p]}
        assert len(values) == 1, ('Conflicting dHash for identical RGB', p)
        hashes[p] = int(next(iter(values)), 16)
    def find(p):
        while parent[p] != p:
            parent[p] = parent[parent[p]]
            p = parent[p]
        return p
    edges = []
    for i, a in enumerate(pixels):
        for b in pixels[:i]:
            distance = (hashes[a] ^ hashes[b]).bit_count()
            if distance <= radius:
                edges.append(dict(left=a, right=b, distance=distance))
                x, y = find(a), find(b)
                if x != y:
                    parent[max(x, y)] = min(x, y)
    grouped = defaultdict(list)
    for p in pixels:
        grouped[find(p)].extend(by_pixel[p])
    components = {min(r['pixel_sha256'] for r in group): sorted(group, key=lambda r: r['sample_id'])
                  for group in grouped.values()}
    return components, edges


def assign_folds(rows, seed=SEED):
    """Source-stratified, largest-group-first balancing of queries AND images."""
    components, edges = image_components(rows)
    buckets = defaultdict(list)
    for cid, group in components.items():
        sources = Counter(r['dataset'] for r in group)
        # A cross-source component stays intact and uses a deterministic bucket.
        source = min(sources, key=lambda s: (-sources[s], s))
        buckets[source].append((cid, group))
    order = sorted(buckets, key=lambda s: (-sum(len(g) for _, g in buckets[s]) / len(buckets[s]), s))
    best = None
    # Structural partition optimization only: no geometry, labels, caches,
    # model predictions or validation outcomes participate in this choice.
    for attempt in range(32):
        sq = {s: [0] * N_FOLDS for s in sorted({r['dataset'] for r in rows})}
        sg = {s: [0] * N_FOLDS for s in sq}
        tq, tg, assignment = [0] * N_FOLDS, [0] * N_FOLDS, {}
        for source in order:
            groups = sorted(buckets[source], key=lambda x: (-len(x[1]), key(seed, 'fold_group', attempt, x[0])))
            for cid, group in groups:
                qcounts = Counter(r['dataset'] for r in group)
                minimum = min(sg[source])
                choices = [f for f in range(N_FOLDS) if sg[source][f] == minimum]
                chosen = min(choices, key=lambda f: (sq[source][f], tg[f], tq[f],
                                                     key(seed, 'fold_tie', attempt, cid, f)))
                assignment[cid] = chosen
                tq[chosen] += len(group)
                tg[chosen] += 1
                for s, n in qcounts.items():
                    sq[s][chosen] += n
                    sg[s][chosen] += 1
        variance = sum((N_FOLDS*n-sum(v))**2 for v in sq.values() for n in v)
        cost = (max(tq)-min(tq), max(tg)-min(tg), variance, digest_json(assignment))
        if best is None or cost < best[0]:
            best = (cost, assignment, tq, tg, sq, sg, attempt)
    _, assignments, total_queries, total_groups, source_queries, source_groups, chosen_attempt = best
    sample_to_fold, image_to_fold, sample_to_group = {}, {}, {}
    for cid, group in components.items():
        f = assignments[cid]
        for row in group:
            sample_to_fold[row['sample_id']] = f
            sample_to_group[row['sample_id']] = cid
            old = image_to_fold.setdefault(row['pixel_sha256'], f)
            assert old == f
    assert len(sample_to_fold) == len(rows)
    assert set(sample_to_fold.values()) == set(range(N_FOLDS))
    return dict(sample_to_fold=sample_to_fold, image_to_fold=image_to_fold,
                sample_to_group=sample_to_group, components=components, near_edges=edges,
                balance=dict(queries=total_queries, components=total_groups,
                             source_queries=source_queries, source_components=source_groups,
                             deterministic_partition_attempts=32, selected_partition_attempt=chosen_attempt,
                             selection_criterion='query range, component range, source-query variance, mapping hash'))


def select_fresh(infos, historical_pixels, historical_hashes, seed=SEED, quota=64):
    """No model outputs are accepted; exclusion is never relaxed to fill quota."""
    tree = HammingIndex(historical_hashes)
    used = set(historical_pixels)
    selected, stats = [], {}
    for slug in DATASETS:
        source = Path('data/public_v1/manifests') / (slug + '.jsonl')
        eligible = [r for r in read_rows(source) if r.get('split', 'test') == 'test'
                    and r.get('expected_actionable') and valid_public_geometry(r)]
        ordered = sorted(eligible, key=lambda r: key(seed, 'fresh_v12', r['sample_id']))
        chosen, rejected = [], Counter()
        for row in ordered:
            info = infos[path_key(row['image'])]
            pixel, dhash = info['pixel_sha256'], int(info['dhash256'], 16)
            if pixel in used:
                rejected['historical_or_selected_exact'] += 1
                continue
            if tree.near(dhash, RADIUS):
                rejected['historical_or_selected_dhash_le4'] += 1
                continue
            chosen.append(dict(row, pixel_sha256=pixel, dhash256=info['dhash256'],
                               image_sha256=info['file_sha256'], experiment_split='test',
                               split='test', training_extension=False))
            used.add(pixel)
            tree.add(dhash)
            if len(chosen) == quota:
                break
        if len(chosen) != quota:
            raise ValueError(f'{slug}: fresh quota {quota} unavailable ({len(chosen)}); do not relax exclusion')
        selected.extend(chosen)
        stats[slug] = dict(selected=len(chosen), eligible_queries=len(eligible), rejected_before_quota=dict(rejected))
    return selected, stats


def authenticated_raw_path(case, source_rows):
    sid = hashlib.sha256(case['sample_id'].encode()).hexdigest()[:24]
    path = Path('runs/semantic_head_v7/features') / (sid + '.npz')
    if not path.is_file():
        path = Path('runs/conditioned_v3/features') / (sid + '.npz')
    assert path.is_file(), ('Missing reusable frozen-Qwen raw features', case['sample_id'])
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive['metadata']))
    for name in ('sample_id', 'image_sha256', 'image_size'):
        assert metadata[name] == case[name], (str(path), name)
    assert metadata['own_token_forward'] and metadata['all_backbone_parameters_frozen']
    assert metadata['ground_truth_in_model_input'] is False
    expected = hashlib.sha256(case['instruction'].encode()).hexdigest()
    if metadata.get('instruction_sha256'):
        assert metadata['instruction_sha256'] == expected
    else:
        manifest = path.parent.parent / 'manifest.jsonl'
        if manifest not in source_rows:
            source_rows[manifest] = {r['sample_id']: r for r in read_rows(manifest)}
        assert source_rows[manifest][case['sample_id']]['instruction'] == case['instruction']
    return str(path.resolve())


def add_roles(rows, mapping):
    source_rows, result = {}, []
    for case in rows:
        train = case['experiment_split'] == 'train'
        f = mapping['sample_to_fold'][case['sample_id']] if train else None
        role = dict(image_fold=f,
                    image_fold_group=mapping['sample_to_group'][case['sample_id']] if train else None,
                    provider='crossfit' if train else 'full',
                    oof_provider=f'fold_{f}' if train else 'full',
                    in_sample_provider=f'fold_{(f + 1) % N_FOLDS}' if train else 'full',
                    raw_path=authenticated_raw_path(case, source_rows)
                    if case['experiment_split'] in ('train', 'val') else None)
        assert not ADDED_FIELDS & set(case), 'Source cases already contain V12 control fields'
        result.append(dict(case, **role))
    return result, list(source_rows)


def make_fold_document(train, mapping, seed):
    all_ids = sorted(r['sample_id'] for r in train)
    all_groups = sorted({r['pixel_sha256'] for r in train})
    roles = {}
    for f in range(N_FOLDS):
        heldout = [r for r in train if r['image_fold'] == f]
        allowed = [r for r in train if r['image_fold'] != f]
        a, h = sorted(r['sample_id'] for r in allowed), sorted(r['sample_id'] for r in heldout)
        assert not set(a) & set(h) and set(a) | set(h) == set(all_ids)
        ag, hg = sorted({r['pixel_sha256'] for r in allowed}), sorted({r['pixel_sha256'] for r in heldout})
        assert not set(ag) & set(hg)
        roles[f'fold_{f}'] = dict(fold=f, allowed_train_ids=a, heldout_ids=h,
            allowed_train_ids_sha256=digest_json(a), heldout_ids_sha256=digest_json(h),
            allowed_train_groups=ag, heldout_groups=hg,
            allowed_train_n=len(a), heldout_n=len(h), allowed_train_rgb_groups=len(ag), heldout_rgb_groups=len(hg),
            allowed_source_queries=dict(Counter(r['dataset'] for r in allowed)),
            heldout_source_queries=dict(Counter(r['dataset'] for r in heldout)),
            heldout_source_rgb_groups=dict(sorted(Counter(s for s, _ in
                {(r['dataset'], r['pixel_sha256']) for r in heldout}).items())))
    roles['full'] = dict(fold=None, allowed_train_ids=all_ids, heldout_ids=[],
        allowed_train_ids_sha256=digest_json(all_ids), heldout_ids_sha256=digest_json([]),
        allowed_train_groups=all_groups, heldout_groups=[], allowed_train_n=len(all_ids), heldout_n=0,
        allowed_train_rgb_groups=len(all_groups), heldout_rgb_groups=0,
        allowed_source_queries=dict(Counter(r['dataset'] for r in train)))
    return dict(schema_version=1, seed=seed, n_folds=N_FOLDS, radius=RADIUS,
        grouping='same canonical RGB or transitive connected component of dHash256 Hamming distance<=4',
        fold_assignment='source-stratified largest-group-first; 32 deterministic structural restarts; no labels or outcomes',
        sample_to_fold=mapping['sample_to_fold'], image_to_fold=mapping['image_to_fold'],
        roles=roles, folds=[roles[f'fold_{f}'] for f in range(N_FOLDS)],
        balance=mapping['balance'], near_edges=mapping['near_edges'],
        train_component_count=len(mapping['components']),
        arm_mapping=dict(oof='fold_image_fold', in_sample='fold_((image_fold+1)%5)', evaluation='full for both arms'),
        initialization_requirement='fresh common initialization; no historical learned readout or supervised head_prior',
        source_group_ids=dict((cid, sorted({r['pixel_sha256'] for r in group})) for cid, group in mapping['components'].items()))


def audit_partition(rows, folds, history_pixels, history_hashes):
    train = [r for r in rows if r['experiment_split'] == 'train']
    val = [r for r in rows if r['experiment_split'] == 'val']
    test = [r for r in rows if r['experiment_split'] == 'test']
    assert Counter(r['experiment_split'] for r in rows) == dict(train=873, val=172, test=192)
    assert len({r['sample_id'] for r in rows}) == len(rows)
    assert len({r['pixel_sha256'] for r in test}) == 192
    assert Counter(r['source'] for r in test) == {s: 64 for s in DATASETS}
    assert all(r['image_fold'] is None and r['provider'] == 'full' and
               r['oof_provider'] == r['in_sample_provider'] == 'full' for r in val + test)
    for r in train:
        f = r['image_fold']
        assert r['sample_id'] not in folds['roles'][f'fold_{f}']['allowed_train_ids']
        assert r['sample_id'] in folds['roles'][f'fold_{(f+1)%5}']['allowed_train_ids']
    near = []
    groups = {r['pixel_sha256']: (r['image_fold'], int(r['dhash256'], 16)) for r in train}
    items = sorted(groups.items())
    for i, (p, (f, h)) in enumerate(items):
        for q, (g, j) in items[:i]:
            if (h ^ j).bit_count() <= RADIUS:
                assert f == g
                near.append((p, q))
    cross = {}
    for name, left, right in [('train_val', train, val), ('train_test', train, test), ('val_test', val, test)]:
        aa = {r['pixel_sha256']: int(r['dhash256'], 16) for r in left}
        bb = {r['pixel_sha256']: int(r['dhash256'], 16) for r in right}
        exact = aa.keys() & bb.keys()
        pairs = sum((a ^ b).bit_count() <= RADIUS for a in aa.values() for b in bb.values())
        assert not exact and not pairs, ('Cross-split leakage', name)
        cross[name] = dict(exact_rgb=0, dhash_le4_pairs=0, comparisons=len(aa)*len(bb))
    tree = HammingIndex(history_hashes)
    for r in test:
        assert r['pixel_sha256'] not in history_pixels
        assert not tree.near(int(r['dhash256'], 16), RADIUS)
        tree.add(int(r['dhash256'], 16))
    return dict(complete=True, counts=dict(Counter(r['experiment_split'] for r in rows)),
        cross_split_overlap=cross, train_near_edges=len(near), train_components=folds['train_component_count'],
        raw_feature_reuse='Qwen NPZ metadata and source instruction authenticated; old learned points/features never reused',
        raw_feature_bytes_rehashed=False, selection_used_predictions=False, all_test_features_at_prepare=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--seed', type=int, default=SEED)
    parser.add_argument('--audit-only', action='store_true')
    args = parser.parse_args()
    os.chdir(ROOT)
    out = args.output.resolve()
    assert out.is_relative_to((ROOT/'runs').resolve()) and out != (ROOT/'runs').resolve()
    assert out != (ROOT/'runs/candidate_scorer_v11').resolve()
    v11_protocol = Path('runs/candidate_scorer_v11/protocol.json')
    assert json.loads(v11_protocol.read_text(encoding='utf-8'))['manifest_sha256'] == sha(BASE)
    base = read_rows(BASE)
    retained = [r for r in base if r['experiment_split'] in ('train', 'val')]
    assert Counter(r['experiment_split'] for r in retained) == dict(train=873, val=172)
    history, ignored = discover_history(ROOT, out)
    infos, pixels, hashes, inputs, history_count = resolve_history(history)
    inputs += [BASE, v11_protocol, Path(__file__), Path('scripts/prepare_candidate_scorer.py'),
               Path('scripts/prepare_semantic_head_data.py')]
    inputs += [Path('data/public_v1/manifests')/(s+'.jsonl') for s in DATASETS]
    source_identity = {str(p): sha(p) for p in sorted(set(map(Path, inputs)))}
    fresh, quota_stats = select_fresh(infos, pixels, hashes, args.seed)
    assignment = assign_folds([r for r in retained if r['experiment_split'] == 'train'], args.seed)
    rows, raw_sources = add_roles(retained + fresh, assignment)
    source_identity.update({str(p): sha(p) for p in raw_sources})
    train = [r for r in rows if r['experiment_split'] == 'train']
    fold_doc = make_fold_document(train, assignment, args.seed)
    fold_doc['source_identity'] = source_identity
    fold_doc['retained_v11_manifest_sha256'] = sha(BASE)
    for old, new in zip(retained, rows):
        assert {k: v for k, v in new.items() if k not in ADDED_FIELDS} == old
    audit = audit_partition(rows, fold_doc, pixels, hashes)
    unique = {}
    for row in rows:
        previous = unique.setdefault(path_key(row['image']), row)
        for field in ('image_sha256', 'pixel_sha256', 'dhash256', 'image_size'):
            assert previous[field] == row[field], ('Conflicting file identity', row['image'], field)
    with ThreadPoolExecutor(max_workers=4) as pool:
        paths = sorted(unique)
        for n, (path, actual) in enumerate(zip(paths, pool.map(image_info, paths)), 1):
            r = unique[path]
            assert r['image_sha256'] == actual['file_sha256'], ('File SHA', path)
            for name in ('pixel_sha256', 'dhash256', 'image_size'):
                assert r[name] == actual[name], (name, path)
            if n % 100 == 0 or n == len(paths):
                print(f'VERIFIED RGB {n}/{len(paths)}', flush=True)
    for path, digest in source_identity.items():
        assert sha(Path(path)) == digest, ('Source changed during preparation', path)
    audit['all_selected_unique_files_rgb_verified'] = len(unique)
    selection = dict(complete=True, seed=args.seed, counts=audit['counts'], quotas=quota_stats,
        selection_sort_key='sha256(seed:fresh_v12:sample_id)', retained_train_val_exact_unchanged=True,
        retained_train_val_source=str(BASE), retained_v11_manifest_sha256=sha(BASE),
        historical_manifest_count=len(history), historical_rows=history_count,
        historical_pixel_count=len(pixels), historical_dhash_count=len(hashes), historical_unresolved=[],
        historical_manifest_sha256={str(p): sha(p) for p in history}, ignored_history_artifact_paths=ignored,
        source_identity=source_identity, near_exclusion_radius=RADIUS,
        near_exclusion_scope='all historical and all pairwise fresh images, across sources',
        selection_used_predictions=False, all_selected_unique_files_rgb_verified=len(unique))
    if args.audit_only or (out/'data_selection.json').exists():
        assert read_rows(out/'manifest.jsonl') == rows
        assert json.loads((out/'folds.json').read_text(encoding='utf-8')) == fold_doc
        saved = json.loads((out/'data_selection.json').read_text(encoding='utf-8'))
        for k, value in selection.items():
            assert saved[k] == value, k
        for split in ('train', 'val', 'test'):
            assert read_rows(out/'data'/f'{split}.jsonl') == [r for r in rows if r['experiment_split'] == split]
        assert saved['manifest_sha256'] == sha(out/'manifest.jsonl')
        assert saved['folds_sha256'] == sha(out/'folds.json')
    else:
        assert not (out/'protocol.json').exists() and not (out/'frozen.json').exists(), 'V12 already sealed'
        out.mkdir(parents=True, exist_ok=True)
        (out/'data').mkdir(exist_ok=True)
        write_rows(out/'manifest.jsonl', rows)
        for split in ('train', 'val', 'test'):
            write_rows(out/'data'/f'{split}.jsonl', [r for r in rows if r['experiment_split'] == split])
        write_json(out/'folds.json', fold_doc)
        selection.update(manifest_sha256=sha(out/'manifest.jsonl'), folds_sha256=sha(out/'folds.json'),
            split_manifest_sha256={s: sha(out/'data'/f'{s}.jsonl') for s in ('train', 'val', 'test')})
        write_json(out/'data_selection.json', selection)
    audit.update(manifest_sha256=sha(out/'manifest.jsonl'), folds_sha256=sha(out/'folds.json'),
                 data_selection_sha256=sha(out/'data_selection.json'), source_identities_unchanged=True)
    if args.audit_only:
        print(json.dumps(dict(audit_complete=True, **audit)), flush=True)
    else:
        write_json(out/'data_selection_audit.json', audit)
        print(json.dumps(dict(prepared=str(out), counts=audit['counts'], fold_balance=assignment['balance'])), flush=True)


if __name__ == '__main__':
    main()
