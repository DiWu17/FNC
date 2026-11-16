"""Offline V12 partition and history-exclusion invariants; no model execution."""
from __future__ import annotations

import copy
import hashlib
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import prepare_crossfit_candidates as prepare


def case(sid, pixel, dhash, source='source_a', split='train'):
    return dict(sample_id=sid, pixel_sha256=pixel, dhash256=f'{dhash:064x}',
                dataset=source, experiment_split=split, instruction='a fixed instruction')


def test_near_duplicate_groups_use_transitive_closure_and_exact_rgb():
    rows = [case('a', 'rgb_a', 0), case('a2', 'rgb_a', 0),
            case('b', 'rgb_b', 15), case('c', 'rgb_c', 255)]
    groups, edges = prepare.image_components(rows)
    assert len(groups) == 1
    assert len(edges) == 2
    assert {r['sample_id'] for r in next(iter(groups.values()))} == {'a', 'a2', 'b', 'c'}


def test_folds_are_deterministic_grouped_source_balanced_and_label_independent():
    rows = []
    for i in range(30):
        pixel = hashlib.sha256(str(i).encode()).hexdigest()
        for j in range(1 + i % 3):
            rows.append(case(f'{i}:{j}', pixel, int(pixel, 16), source=f'source_{i % 2}'))
    first = prepare.assign_folds(rows)
    changed = copy.deepcopy(list(reversed(rows)))
    for r in changed:
        r['instruction'] = 'different wording'
        r['target_geometry'] = {'type': 'bbox', 'boxes': [[0, 0, 1, 1]]}
    second = prepare.assign_folds(changed)
    assert first['sample_to_fold'] == second['sample_to_fold']
    assert len(first['image_to_fold']) == 30
    for values in first['balance']['source_components'].values():
        assert max(values) - min(values) <= 1
    for r in rows:
        assert first['sample_to_fold'][r['sample_id']] == first['image_to_fold'][r['pixel_sha256']]


def test_role_lists_enforce_oof_exclusion_and_is_inclusion(monkeypatch):
    rows = []
    for i in range(10):
        pixel = hashlib.sha256(str(i).encode()).hexdigest()
        rows.append(case(str(i), pixel, int(pixel, 16)))
    assignment = prepare.assign_folds(rows)
    monkeypatch.setattr(prepare, 'authenticated_raw_path', lambda row, sources: '/frozen/raw.npz')
    annotated, _ = prepare.add_roles(rows, assignment)
    document = prepare.make_fold_document(annotated, assignment, 20261201)
    assert set(document['roles']) == {'fold_0', 'fold_1', 'fold_2', 'fold_3', 'fold_4', 'full'}
    assert len(document['roles']['full']['allowed_train_ids']) == 10
    for row in annotated:
        f, sid = row['image_fold'], row['sample_id']
        oof = document['roles'][f'fold_{f}']
        ins = document['roles'][f'fold_{(f+1)%5}']
        assert sid in oof['heldout_ids'] and sid not in oof['allowed_train_ids']
        assert sid in ins['allowed_train_ids'] and sid not in ins['heldout_ids']
        assert not set(oof['allowed_train_groups']) & set(oof['heldout_groups'])


def test_validation_and_test_share_full_provider_with_no_training_fold(monkeypatch):
    rows = [case('v', 'v_rgb', 1, split='val'), case('t', 't_rgb', 2, split='test')]
    monkeypatch.setattr(prepare, 'authenticated_raw_path', lambda row, sources: '/frozen/raw.npz')
    result, _ = prepare.add_roles(rows, {'sample_to_fold': {}, 'sample_to_group': {}})
    assert all(r['image_fold'] is None and r['provider'] == 'full' for r in result)
    assert all(r['oof_provider'] == r['in_sample_provider'] == 'full' for r in result)
    assert result[0]['raw_path'] == '/frozen/raw.npz'
    assert result[1]['raw_path'] is None


def test_fresh_selection_never_relaxes_historical_or_pairwise_exclusion(monkeypatch, tmp_path):
    infos, sources = {}, {}
    hashes = ['f0' * 32, '0f' * 32, 'aa' * 32]
    for ds, dhash in zip(prepare.DATASETS, hashes):
        image = str(tmp_path / (ds + '.png'))
        infos[prepare.path_key(image)] = dict(pixel_sha256=ds, dhash256=dhash, file_sha256='file')
        sources[ds] = [dict(sample_id=ds, image=image, expected_actionable=True,
                            target_geometry={'type': 'bbox', 'boxes': [[0, 0, 1, 1]]})]
    monkeypatch.setattr(prepare, 'read_rows', lambda path: sources[Path(path).stem])
    rows, stats = prepare.select_fresh(infos, {'old_rgb'}, {0}, quota=1)
    assert len(rows) == 3
    assert all(s['selected'] == 1 for s in stats.values())
    with pytest.raises(ValueError, match='do not relax exclusion'):
        prepare.select_fresh(infos, {'old_rgb', prepare.DATASETS[0]}, {0}, quota=1)


def test_rgb_identity_conflicts_fail_closed():
    with pytest.raises(AssertionError, match='Conflicting dHash'):
        prepare.image_components([case('a', 'same', 1), case('b', 'same', 2)])
