"""V13 data-only preparation, with sealed V12 tools and strict history exclusion.

Primary: ScreenSpot-Pro96 + VenusBench-GD96. The six remaining ScreenSpot-v2
images are a separately reported supplement. No model or prediction is loaded.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path

from prepare_semantic_head_data import (HammingIndex, discover_history, image_info,
    key, read_json, read_rows, sha, valid_public_geometry, write_json, write_rows)
from prepare_candidate_scorer import resolve_history, path_key
from prepare_crossfit_candidates import image_components

ROOT = Path(__file__).resolve().parents[1]
OUT = Path('runs/multiregion_v13')
V12 = Path('runs/crossfit_candidates_v12')
SEED = 20261301
RADIUS = 4
PRIMARY_QUOTAS = {'screenspot_pro': 96, 'venusbench_gd': 96}
SUPPLEMENT = 'screenspot_v2'
SUPPLEMENT_N = 6
SOURCES = (SUPPLEMENT, *PRIMARY_QUOTAS)
TOOLS = ('scripts/prepare_semantic_head_data.py', 'scripts/prepare_candidate_scorer.py',
         'scripts/prepare_crossfit_candidates.py')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def seal_inventory(seal):
    entries = {}
    for name in ('files', 'training_evidence'):
        for path, expected in seal[name].items():
            normalized = path_key(path)
            prior = entries.setdefault(normalized, dict(sha256=expected, references=[]))
            assert prior['sha256'] == expected, ('Conflicting V12 sealed identity', path)
            prior['references'].append(dict(inventory=name, path=path))
    return entries


def authenticate(path, inventory):
    path = Path(path)
    expected = inventory.get(path_key(path))
    assert expected is not None, ('Input is not pinned by V12', str(path))
    actual = sha(path)
    assert actual == expected['sha256'], ('Sealed V12 input changed', str(path))
    return dict(path=str(path.resolve()), sha256=actual, matches_v12_seal=True,
                legacy_references=expected['references'])


def remaining_candidates(infos, history_pixels, history_hashes, source_rows):
    """Availability before new-cohort mutual exclusion; never reads scores."""
    tree = HammingIndex(history_hashes)
    available, summary = {}, {}
    for slug, rows in source_rows.items():
        eligible = [r for r in rows if r.get('split', 'test') == 'test'
                    and r.get('expected_actionable') and valid_public_geometry(r)]
        retained, rejected = [], Counter()
        for row in eligible:
            info = infos[path_key(row['image'])]
            pixel, value = info['pixel_sha256'], int(info['dhash256'], 16)
            if pixel in history_pixels:
                rejected['historical_exact'] += 1
            elif tree.near(value, RADIUS):
                rejected['historical_near_le4'] += 1
            else:
                retained.append(dict(row, pixel_sha256=pixel, dhash256=info['dhash256'],
                                     image_sha256=info['file_sha256']))
        components, edges = image_components(retained)
        available[slug] = retained
        summary[slug] = dict(eligible_queries=len(eligible), remaining_queries=len(retained),
            remaining_unique_rgb=len({r['pixel_sha256'] for r in retained}),
            remaining_near_components=len(components), remaining_near_edges=len(edges),
            rejected_queries=dict(rejected))
    return available, summary


def select_cohorts(available, history_pixels, history_hashes, seed=SEED,
                   primary_quotas=None, supplement_n=SUPPLEMENT_N):
    """Reserve the exhausted source first; all new images share one BK tree."""
    quotas = dict(PRIMARY_QUOTAS if primary_quotas is None else primary_quotas)
    tree, used = HammingIndex(history_hashes), set(history_pixels)
    stats = {}
    def take(slug, quota, split, take_all=False):
        selected, rejected = [], Counter()
        for row in sorted(available[slug], key=lambda r: key(seed, 'fresh_v13', r['sample_id'])):
            pixel, value = row['pixel_sha256'], int(row['dhash256'], 16)
            if pixel in used:
                rejected['exact'] += 1
                continue
            if tree.near(value, RADIUS):
                rejected['near_le4'] += 1
                continue
            selected.append(dict(row, source_split=row.get('source_split', row.get('split', 'test')),
                experiment_split=split, split=split, training_extension=False,
                image_fold=None, image_fold_group=None, provider='full', oof_provider='full',
                in_sample_provider='full', raw_path=None))
            used.add(pixel)
            tree.add(value)
            if not take_all and len(selected) == quota:
                break
        if len(selected) != quota:
            raise ValueError(f'{slug}: required {quota} strictly fresh {split} images, found {len(selected)}; exclusion is not relaxed')
        stats[slug] = dict(selected=len(selected), split=split, mutual_exclusion_rejected=dict(rejected))
        return selected
    supplement = take(SUPPLEMENT, supplement_n, 'supplement', take_all=True)
    primary = []
    for slug, quota in quotas.items():
        primary.extend(take(slug, quota, 'test'))
    assert len({r['pixel_sha256'] for r in primary + supplement}) == len(primary) + len(supplement)
    return primary, supplement, stats


def leave_source_validation(train, val, proposer_roles):
    """Source-heldout scorer CV for epoch selection, not an unbiased endpoint."""
    definitions = {}
    sources = sorted({r['dataset'] for r in train})
    assert set(sources) == {r['dataset'] for r in val}
    for source in sources:
        allowed = sorted(r['sample_id'] for r in train if r['dataset'] != source)
        withheld = sorted(r['sample_id'] for r in train if r['dataset'] == source)
        cv_val = sorted(r['sample_id'] for r in val if r['dataset'] == source)
        assert not set(allowed) & set(withheld)
        exposure = {role: info['allowed_source_queries'].get(source, 0)
                    for role, info in proposer_roles.items()}
        assert exposure and all(n > 0 for n in exposure.values()), ('Expected upstream source exposure', source)
        definitions[source] = dict(gradient_train_ids=allowed, withheld_train_ids=withheld, cv_validation_ids=cv_val,
            gradient_train_n=len(allowed), withheld_train_n=len(withheld), cv_validation_n=len(cv_val),
            gradient_train_ids_sha256=digest(allowed), withheld_train_ids_sha256=digest(withheld),
            cv_validation_ids_sha256=digest(cv_val),
            proposer_training_queries_from_heldout_source_by_role=exposure,
            frozen_proposer_has_seen_heldout_source=True,
            original_validation_was_used_for_upstream_checkpoint_selection=True)
    return dict(scope='scorer-only source-heldout cross-validation for common-epoch selection',
        definitions=definitions, source_count=len(sources),
        gradient_rule='No train-h or val-h labels enter the fold scorer gradient; non-h original training only',
        cv_rule='For each of the two prespecified losses, train four source folds for eight epochs; evaluate each val-h every epoch',
        epoch_selection='Use the four-source macro average at a common epoch for S1; apply that epoch to full873 trajectory; S2 inherits S1 epoch',
        heldout_validation_labels_allowed_for_cv_selection=True,
        cv_score_is_independent_final_evaluation=False,
        final_independent_evaluation='fresh Pro96 plus Venus96 only; separate v2 supplement is not pooled',
        limitation='Frozen proposers have seen all four sources and used original172 validation for checkpoint selection; this is not full-pipeline unseen-domain evaluation')


def verify_independence(train, val, primary, supplement, history_pixels, history_hashes):
    fresh = primary + supplement
    tree, used = HammingIndex(history_hashes), set(history_pixels)
    for row in fresh:
        assert row['pixel_sha256'] not in used, ('Historical/new exact reuse', row['sample_id'])
        assert not tree.near(int(row['dhash256'], 16), RADIUS), ('Historical/new near reuse', row['sample_id'])
        used.add(row['pixel_sha256'])
        tree.add(int(row['dhash256'], 16))
    groups = {'train': train, 'val': val, 'test': primary, 'supplement': supplement}
    cross = {}
    for i, (left_name, left) in enumerate(groups.items()):
        aa = {r['pixel_sha256']: int(r['dhash256'], 16) for r in left}
        for right_name, right in list(groups.items())[i+1:]:
            bb = {r['pixel_sha256']: int(r['dhash256'], 16) for r in right}
            assert not aa.keys() & bb.keys(), (left_name, right_name, 'exact leakage')
            nnear = sum((a ^ b).bit_count() <= RADIUS for a in aa.values() for b in bb.values())
            assert nnear == 0, (left_name, right_name, 'near leakage')
            cross[f'{left_name}__{right_name}'] = dict(exact_rgb=0, dhash_le4_pairs=0, comparisons=len(aa)*len(bb))
    return cross


def copy_frozen_bytes(source, destination):
    payload = Path(source).read_bytes()
    destination = Path(destination)
    if destination.exists():
        assert destination.read_bytes() == payload, ('Existing V13 retained cohort differs', str(destination))
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix('.jsonl.tmp')
    temporary.write_bytes(payload)
    temporary.replace(destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUT)
    parser.add_argument('--seed', type=int, default=SEED)
    parser.add_argument('--audit-only', action='store_true')
    args = parser.parse_args()
    os.chdir(ROOT)
    out = args.output.resolve()
    assert out.is_relative_to((ROOT/'runs').resolve()) and out != (ROOT/'runs').resolve()
    assert not any(out == (ROOT/'runs'/name).resolve() for name in ('candidate_scorer_v11', 'crossfit_candidates_v12'))
    seal_path = V12/'frozen.json'
    sealed = read_json(seal_path)
    assert sealed['complete']
    inventory = seal_inventory(sealed)
    identity = {str(Path(p).resolve()): authenticate(p, inventory) for p in TOOLS}
    train_path, val_path, folds_path = V12/'data/train.jsonl', V12/'data/val.jsonl', V12/'folds.json'
    for p in (train_path, val_path, folds_path):
        identity[str(p.resolve())] = authenticate(p, inventory)
    train, val = read_rows(train_path), read_rows(val_path)
    assert len(train) == 873 and len(val) == 172
    history, ignored = discover_history(ROOT, out)
    assert not any('predictions' in {p.lower() for p in path.parts} for path in history), 'Use metadata manifests, never prediction content'
    infos, pixels, hashes, inputs, history_rows = resolve_history(history)
    public_paths = {s: Path('data/public_v1/manifests')/(s+'.jsonl') for s in SOURCES}
    for p in sorted(set(map(Path, inputs)) | set(public_paths.values())):
        identity[str(p.resolve())] = authenticate(p, inventory)
    script_hash = sha(__file__)
    seal_hash = sha(seal_path)
    available, availability = remaining_candidates(infos, pixels, hashes,
        {slug: read_rows(path) for slug, path in public_paths.items()})
    print(json.dumps(dict(stage='remaining_candidates', availability=availability)), flush=True)
    try:
        primary, supplement, quota_stats = select_cohorts(available, pixels, hashes, args.seed)
    except ValueError as error:
        failure = dict(complete=False, reason=str(error), availability=availability,
            seed=args.seed, desired_primary_quotas=PRIMARY_QUOTAS, desired_supplement=SUPPLEMENT_N,
            exclusion_relaxed=False, no_partial_test_manifest_published=True)
        if not args.audit_only:
            write_json(out/'data_selection_failure.json', failure)
        print(json.dumps(failure), flush=True)
        raise SystemExit(2) from error
    rows = train + val + primary + supplement
    assert len({r['sample_id'] for r in rows}) == len(rows)
    assert Counter(r['experiment_split'] for r in rows) == dict(train=873, val=172, test=192, supplement=6)
    assert Counter(r['source'] for r in primary) == PRIMARY_QUOTAS
    cross = verify_independence(train, val, primary, supplement, pixels, hashes)
    loso = leave_source_validation(train, val, read_json(folds_path)['roles'])
    unique = {}
    for r in rows:
        previous = unique.setdefault(path_key(r['image']), r)
        for name in ('image_sha256', 'pixel_sha256', 'dhash256', 'image_size'):
            assert previous[name] == r[name]
    with ThreadPoolExecutor(max_workers=4) as pool:
        paths = sorted(unique)
        for n, (path, actual) in enumerate(zip(paths, pool.map(image_info, paths)), 1):
            row = unique[path]
            assert row['image_sha256'] == actual['file_sha256']
            for name in ('pixel_sha256', 'dhash256', 'image_size'):
                assert row[name] == actual[name], (name, path)
            if n % 100 == 0 or n == len(paths):
                print(f'VERIFIED RGB {n}/{len(paths)}', flush=True)
    for path, entry in identity.items():
        assert sha(path) == entry['sha256'], ('Input changed during prepare', path)
    assert sha(__file__) == script_hash and sha(seal_path) == seal_hash
    selection = dict(complete=True, version=13, seed=args.seed, selection_namespace='fresh_v13',
        primary_test_n=192, primary_quotas=PRIMARY_QUOTAS, supplement_source=SUPPLEMENT, supplement_n=6,
        counts=dict(Counter(r['experiment_split'] for r in rows)), availability=availability, quota_stats=quota_stats,
        quota_rationale='Original64x3 impossible: only6 untouched v2 images; prespecified Pro96+Venus96 primary and separate v2 supplement',
        selection_order=['reserve_all_screenspot_v2_supplement', *PRIMARY_QUOTAS],
        selection_sort_key='sha256(seed:fresh_v13:sample_id)',
        historical_manifest_count=len(history), historical_rows=history_rows,
        historical_unique_rgb=len(pixels), historical_unique_dhash=len(hashes), historical_unresolved=[],
        historical_manifest_sha256={str(p): sha(p) for p in history}, ignored_history_artifact_paths=ignored,
        source_identity=identity, v12_frozen_path=str(seal_path), v12_frozen_sha256=seal_hash,
        preparation_source_path=str(Path(__file__).resolve()), preparation_source_sha256=script_hash,
        exclusion_radius=RADIUS, exclusion_scope='all history and all pairwise fresh primary/supplement images across sources',
        selection_used_predictions=False, no_model_execution=True,
        retained_train_val_byte_identical_to_v12=True, all_selected_unique_files_rgb_verified=len(unique),
        fresh_gpu_extraction_allowed_only_after_scorer_freeze=True,
        old_v12_test_used_as_migration_diagnostic=False,
        primary_scores_not_directly_comparable_to_historical_64x3_cohorts=True)
    audit = dict(complete=True, cross_split_overlap=cross, train_val_byte_identical=True,
        all_selected_unique_files_rgb_verified=len(unique), fresh_unique_rgb=len({r['pixel_sha256'] for r in primary+supplement}),
        historical_unresolved=0, test_predictions_read=False, no_gpu_or_model_execution=True,
        source_identities_unchanged=True, primary_denominator=192, supplement_denominator=6,
        scorer_cv_scope='source-heldout CV is used for epoch selection, not an independent endpoint')
    datasets = dict(train=train, val=val, test=primary, supplement=supplement)
    existing = args.audit_only or (out/'data_selection.json').exists()
    if existing:
        assert read_rows(out/'manifest.jsonl') == rows
        saved = read_json(out/'data_selection.json')
        for name, value in selection.items():
            assert saved[name] == value, name
        assert read_json(out/'data/leave_source_validation.json') == loso
        for split, cases in datasets.items():
            assert read_rows(out/'data'/f'{split}.jsonl') == cases
        assert sha(out/'data/train.jsonl') == sha(train_path)
        assert sha(out/'data/val.jsonl') == sha(val_path)
        for path, digest_value in saved['artifact_sha256'].items():
            assert sha(path) == digest_value, path
        print(json.dumps(dict(audit_complete=True, data_selection_sha256=sha(out/'data_selection.json'))), flush=True)
        return
    assert not (out/'frozen.json').exists(), 'V13 already frozen'
    (out/'data').mkdir(parents=True, exist_ok=True)
    copy_frozen_bytes(train_path, out/'data/train.jsonl')
    copy_frozen_bytes(val_path, out/'data/val.jsonl')
    write_rows(out/'data/test.jsonl', primary)
    write_rows(out/'data/supplement.jsonl', supplement)
    write_rows(out/'manifest.jsonl', rows)
    write_json(out/'data/leave_source_validation.json', loso)
    artifacts = [out/'manifest.jsonl', out/'data/leave_source_validation.json']
    artifacts += [out/'data'/f'{s}.jsonl' for s in datasets]
    selection['artifact_sha256'] = {str(p): sha(p) for p in artifacts}
    selection['manifest_sha256'] = sha(out/'manifest.jsonl')
    write_json(out/'data_selection.json', selection)
    audit.update(manifest_sha256=selection['manifest_sha256'],
        data_selection_sha256=sha(out/'data_selection.json'), artifact_sha256=selection['artifact_sha256'])
    write_json(out/'data_selection_audit.json', audit)
    print(json.dumps(dict(complete=True, output=str(out), counts=selection['counts'],
                         manifest_sha256=selection['manifest_sha256'])), flush=True)


if __name__ == '__main__':
    main()
