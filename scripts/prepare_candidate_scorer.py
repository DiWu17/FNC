"""CPU-only v11 selection: immutable train/val and history-excluded fresh test.

No predictions, model execution, protocol creation, or historical file writes.
The v9 test is copied separately as an already-analyzed diagnostic cohort.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import time

from prepare_semantic_head_data import (
    HammingIndex, discover_history, image_info, key, read_json, read_rows,
    sha, valid_public_geometry, write_json, write_rows,
)


ROOT = Path(__file__).resolve().parents[1]
OUT = Path('runs/candidate_scorer_v11')
BASE = Path('runs/semantic_head_v7/manifest.jsonl')
DIAGNOSTIC_SOURCE = Path('runs/context_verifier_v9/manifest.jsonl')
PUBLIC_INDEX = Path('data/conditioned_v3/public_image_index.json')
TRAINING_POOL = Path('data/conditioned_v3/training_pool_b8ed555f5e93.jsonl')
OLD_HASH_CACHE = Path('runs/semantic_head_v7/data_selection_image_hash_cache.jsonl')
DATASETS = ('screenspot_v2', 'screenspot_pro', 'venusbench_gd')
SEED = 20261131
RADIUS = 4


def path_key(value, manifest=None):
    path = Path(value)
    if not path.is_absolute() and manifest is not None:
        path = Path(manifest).parent / path
    return os.path.normcase(os.path.abspath(path))


def checked_hashes(paths):
    return {str(p): sha(p) for p in sorted(set(map(Path, paths)))}


def assert_hashes(identities):
    for path, digest in identities.items():
        assert sha(path) == digest, ('Input changed', path)


def resolve_history(history):
    """Same canonical RGB/dHash lookup and authenticated control fallback as v8."""
    index = read_json(PUBLIC_INDEX)
    assert index['complete'] is True
    infos = {path_key(r['image']): r for r in index['images']}
    for path in (TRAINING_POOL, OLD_HASH_CACHE):
        for row in read_rows(path):
            infos.setdefault(path_key(row['image']), row)
    pixels, hashes, unresolved, controls = set(), set(), [], {}
    identity_paths = [PUBLIC_INDEX, TRAINING_POOL, OLD_HASH_CACHE, *history]
    row_count = 0
    for manifest in history:
        for row in read_rows(manifest):
            row_count += 1
            if not row.get('image') and not row.get('image_path'):
                continue
            info = row if row.get('pixel_sha256') and row.get('dhash256') else infos.get(
                path_key(row.get('image', row.get('image_path')), manifest))
            if info is None:
                control = manifest.parent / 'control.json'
                if control.exists():
                    if control not in controls:
                        identity_paths.append(control)
                        cr = read_json(control)
                        original = Path(cr.get('source_manifest', ''))
                        if original.is_file() and sha(original) == cr.get('source_manifest_sha256'):
                            identity_paths.append(original)
                            controls[control] = (original, {r['sample_id']: r for r in read_rows(original)})
                        else:
                            controls[control] = None
                    cached = controls[control]
                    if cached is not None:
                        original, source_rows = cached
                        source = source_rows.get(row.get('sample_id'))
                        if source:
                            info = source if source.get('pixel_sha256') and source.get('dhash256') else infos.get(
                                path_key(source.get('image', source.get('image_path')), original))
            if info is None:
                unresolved.append(dict(manifest=str(manifest), sample_id=row.get('sample_id')))
                continue
            pixels.add(info['pixel_sha256'])
            hashes.add(int(info['dhash256'], 16))
    assert not unresolved, ('Unresolved historical images; exclusion not weakened', unresolved[:20])
    return infos, pixels, hashes, identity_paths, row_count


def split_overlap(left, right):
    """Exact pairwise audit over unique RGB images, including near duplicates."""
    a = {r['pixel_sha256']: int(r['dhash256'], 16) for r in left}
    b = {r['pixel_sha256']: int(r['dhash256'], 16) for r in right}
    exact = sorted(a.keys() & b.keys())
    near = [(pa, pb, (ha ^ hb).bit_count()) for pa, ha in a.items()
            for pb, hb in b.items() if (ha ^ hb).bit_count() <= RADIUS]
    assert not exact and not near, ('Cross-split exact/near leakage', exact[:5], near[:5])
    return dict(left_unique_images=len(a), right_unique_images=len(b),
                exact_pixel_overlaps=0, dhash_le_4_pairs=0,
                comparisons=len(a) * len(b))


def verify_images(rows):
    """Decode every selected/retained unique file; no metadata-only fresh audit."""
    by_path = {}
    for row in rows:
        path = path_key(row['image'])
        previous = by_path.setdefault(path, row)
        for name in ('image_sha256', 'pixel_sha256', 'dhash256', 'image_size'):
            assert previous[name] == row[name], ('Conflicting image identity', path, name)
    ordered = sorted(by_path)
    with ThreadPoolExecutor(max_workers=4) as pool:
        for n, (path, actual) in enumerate(zip(ordered, pool.map(image_info, ordered)), 1):
            row = by_path[path]
            assert row['image_sha256'] == actual['file_sha256'], ('File hash', path)
            for name in ('pixel_sha256', 'dhash256', 'image_size'):
                assert row[name] == actual[name], (name, path)
            if n % 100 == 0 or n == len(ordered):
                print(f'VERIFIED RGB {n}/{len(ordered)}', flush=True)
    return len(ordered)


def audit_rows(manifest, diagnostic, retained, original_diagnostic, pixels, hashes):
    assert [r for r in manifest if r['experiment_split'] in ('train', 'val')] == retained
    assert Counter(r['experiment_split'] for r in manifest) == dict(train=873, val=172, test=192)
    assert len({r['sample_id'] for r in manifest}) == len(manifest)
    assert len(diagnostic) == 192
    assert not {r['sample_id'] for r in manifest} & {r['sample_id'] for r in diagnostic}
    for row, source in zip(diagnostic, original_diagnostic):
        expected = dict(source, experiment_split='diagnostic', split='diagnostic',
                        diagnostic_reuse=True, diagnostic_source_experiment='context_verifier_v9',
                        allowed_use='secondary_diagnostic_only_not_training_or_epoch_selection')
        assert row == expected
    fresh = [r for r in manifest if r['experiment_split'] == 'test']
    assert Counter(r['dataset'] for r in fresh) == Counter(r['dataset'] for r in original_diagnostic)
    assert len({r['pixel_sha256'] for r in fresh}) == 192
    tree = HammingIndex(hashes)
    for row in fresh:
        assert row['pixel_sha256'] not in pixels
        assert not tree.near(int(row['dhash256'], 16), RADIUS), ('Historical/fresh near image', row['sample_id'])
        tree.add(int(row['dhash256'], 16))
    cohorts = {s: [r for r in manifest if r['experiment_split'] == s] for s in ('train', 'val', 'test')}
    cohorts['diagnostic'] = diagnostic
    names = list(cohorts)
    comparisons = {f'{a}__{b}': split_overlap(cohorts[a], cohorts[b])
                   for i, a in enumerate(names) for b in names[i + 1:]}
    return comparisons


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit-only', action='store_true')
    args = parser.parse_args()
    os.chdir(ROOT)
    start = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    history, ignored = discover_history(ROOT, OUT)
    infos, pixels, hashes, identities, history_rows = resolve_history(history)
    identities += [BASE, DIAGNOSTIC_SOURCE, Path(__file__)]
    public_sources = [Path('data/public_v1/manifests') / (slug + '.jsonl') for slug in DATASETS]
    identities += public_sources
    source_hashes = checked_hashes(identities)
    print(f'HISTORY {len(history)} manifests, {len(pixels)} pixels, {len(hashes)} dHashes; unresolved=0', flush=True)
    retained = [r for r in read_rows(BASE) if r['experiment_split'] in ('train', 'val')]
    assert Counter(r['experiment_split'] for r in retained) == dict(train=873, val=172)
    # Verify v8/v9 really retain the same untouched rows before reusing features.
    for previous in ('runs/point_aligned_v8/manifest.jsonl', str(DIAGNOSTIC_SOURCE)):
        assert [r for r in read_rows(previous) if r['experiment_split'] in ('train', 'val')] == retained
    original_diagnostic = [r for r in read_rows(DIAGNOSTIC_SOURCE) if r['experiment_split'] == 'test']
    diagnostic = [dict(r, experiment_split='diagnostic', split='diagnostic', diagnostic_reuse=True,
                       diagnostic_source_experiment='context_verifier_v9',
                       allowed_use='secondary_diagnostic_only_not_training_or_epoch_selection') for r in original_diagnostic]
    selected_counts = {}
    tree, used_pixels, fresh = HammingIndex(hashes), set(pixels), []
    for slug, source in zip(DATASETS, public_sources):
        eligible = [r for r in read_rows(source) if r.get('split', 'test') == 'test'
                    and r.get('expected_actionable') and valid_public_geometry(r)]
        normalized = []
        for row in eligible:
            info = infos[path_key(row['image'])]
            normalized.append(dict(row, pixel_sha256=info['pixel_sha256'], dhash256=info['dhash256'],
                                   image_sha256=info['file_sha256']))
        normalized.sort(key=lambda r: key(SEED, 'fresh_v11', r['sample_id']))
        chosen, rejected = [], Counter()
        for row in normalized:
            if row['pixel_sha256'] in used_pixels:
                rejected['exact'] += 1
                continue
            if tree.near(int(row['dhash256'], 16), RADIUS):
                rejected['near4'] += 1
                continue
            chosen.append(dict(row, experiment_split='test', split='test', training_extension=False))
            used_pixels.add(row['pixel_sha256'])
            tree.add(int(row['dhash256'], 16))
            if len(chosen) == 64:
                break
        assert len(chosen) == 64, ('Insufficient fresh images; exclusion not weakened', slug, len(chosen))
        fresh.extend(chosen)
        selected_counts[slug] = dict(selected=64, eligible_queries=len(eligible), rejected_before_quota=dict(rejected))
        print(f'SELECTED {slug}: 64', flush=True)
    manifest = retained + fresh
    comparisons = audit_rows(manifest, diagnostic, retained, original_diagnostic, pixels, hashes)
    if args.audit_only or (OUT / 'data_selection.json').exists():
        assert read_rows(OUT / 'manifest.jsonl') == manifest
        assert read_rows(OUT / 'diagnostic_manifest.jsonl') == diagnostic
        saved = read_json(OUT / 'data_selection.json')
        assert saved['complete'] and saved['seed'] == SEED
        assert saved['manifest_sha256'] == sha(OUT / 'manifest.jsonl')
        assert saved['diagnostic_manifest_sha256'] == sha(OUT / 'diagnostic_manifest.jsonl')
        assert saved['source_sha256'] == source_hashes, 'Source identities changed since selection'
    images_verified = verify_images(manifest + diagnostic)
    assert_hashes(source_hashes)
    if not args.audit_only and not (OUT / 'data_selection.json').exists():
        write_rows(OUT / 'manifest.jsonl', manifest)
        write_rows(OUT / 'diagnostic_manifest.jsonl', diagnostic)
        write_json(OUT / 'data_selection.json', dict(
            complete=True, seed=SEED, selection_sort_key='sha256(seed:fresh_v11:sample_id)',
            counts=dict(Counter(r['experiment_split'] for r in manifest)), quotas=selected_counts,
            retained_train_val_source=str(BASE), retained_source_sha256=sha(BASE),
            retained_train_val_exact_unchanged=True, source_sha256=source_hashes,
            historical_manifest_sha256={str(p): sha(p) for p in history},
            ignored_history_artifact_paths=ignored, historical_manifest_count=len(history),
            historical_rows=history_rows, historical_pixel_count=len(pixels), historical_dhash_count=len(hashes),
            historical_unresolved=[], public_manifest_sha256={str(p): sha(p) for p in public_sources},
            fresh_pixels_verified=192, all_cohort_unique_files_rgb_verified=images_verified,
            near_exclusion_radius=RADIUS, near_exclusion_scope='all_historical_and_pairwise_fresh',
            cross_split_overlap=comparisons, selection_used_predictions=False,
            diagnostic=dict(source=str(DIAGNOSTIC_SOURCE), source_sha256=sha(DIAGNOSTIC_SOURCE), count=192,
                            previously_analyzed=True, allowed_use='secondary_diagnostic_only_not_training_or_epoch_selection'),
            manifest_sha256=sha(OUT / 'manifest.jsonl'),
            diagnostic_manifest_sha256=sha(OUT / 'diagnostic_manifest.jsonl')))
    write_json(OUT / 'data_selection_audit.json', dict(
        complete=True, manifest_sha256=sha(OUT / 'manifest.jsonl'),
        diagnostic_manifest_sha256=sha(OUT / 'diagnostic_manifest.jsonl'),
        data_selection_sha256=sha(OUT / 'data_selection.json'),
        all_cohort_unique_files_rgb_verified=images_verified, cross_split_overlap=comparisons,
        historical_unresolved=0, source_identities_unchanged=True, seconds=round(time.time() - start, 3)))
    print(f'COMPLETE {OUT / "manifest.jsonl"} ({time.time() - start:.1f}s)', flush=True)


if __name__ == '__main__':
    main()
