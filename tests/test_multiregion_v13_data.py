"""V13 metadata-only selection, strict exclusion and scorer-CV scope tests."""
import hashlib
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import prepare_multiregion_v13 as prepare


def row(name, source, value=None):
    pixel = hashlib.sha256(name.encode()).hexdigest()
    return dict(sample_id=name, source=source, dataset=source, pixel_sha256=pixel,
                dhash256=f'{int(pixel, 16) if value is None else value:064x}',
                split='test', image='/not-opened/' + name, expected_actionable=True,
                target_geometry={'type': 'bbox', 'boxes': [[0, 0, 1, 1]]})


def available():
    return {source: [row(source + str(i), source) for i in range(4)]
            for source in prepare.SOURCES}


def test_original_quota_is_not_silently_relaxed():
    data = available()
    with pytest.raises(ValueError, match='exclusion is not relaxed'):
        prepare.select_cohorts(data, set(), set(), primary_quotas={'screenspot_pro': 1, 'venusbench_gd': 1},
                               supplement_n=64)


def test_supplement_is_reserved_and_primary_cannot_use_a_near_image():
    data = available()
    data['screenspot_v2'] = data['screenspot_v2'][:1]
    protected = int(data['screenspot_v2'][0]['dhash256'], 16)
    near = row('different-rgb-near-supplement', 'screenspot_pro', protected ^ 1)
    data['screenspot_pro'].append(near)
    primary, supplement, _ = prepare.select_cohorts(data, set(), set(),
        primary_quotas={'screenspot_pro': 2, 'venusbench_gd': 2}, supplement_n=1)
    assert len(primary) == 4 and len(supplement) == 1
    assert near['sample_id'] not in {r['sample_id'] for r in primary}
    assert all(r['experiment_split'] == 'test' and r['provider'] == 'full' and r['raw_path'] is None for r in primary)
    assert supplement[0]['experiment_split'] == 'supplement'
    changed_order = {s: list(reversed(v)) for s, v in data.items()}
    again = prepare.select_cohorts(changed_order, set(), set(),
        primary_quotas={'screenspot_pro': 2, 'venusbench_gd': 2}, supplement_n=1)
    assert (primary, supplement) == again[:2]


def test_availability_counts_queries_rgb_and_near_exclusion():
    good = row('good', 'screenspot_v2')
    duplicate = dict(good, sample_id='good-query2')
    old = row('old', 'screenspot_v2')
    near = row('near', 'screenspot_v2', 1)
    candidates = [good, duplicate, old, near]
    infos = {prepare.path_key(r['image']): dict(pixel_sha256=r['pixel_sha256'], dhash256=r['dhash256'], file_sha256='file')
             for r in candidates}
    rows, stats = prepare.remaining_candidates(infos, {old['pixel_sha256']}, {0}, {'screenspot_v2': candidates})
    assert len(rows['screenspot_v2']) == 2
    assert stats['screenspot_v2']['remaining_unique_rgb'] == 1
    assert stats['screenspot_v2']['remaining_near_components'] == 1
    assert stats['screenspot_v2']['rejected_queries'] == {'historical_exact': 1, 'historical_near_le4': 1}


def test_source_cv_h_labels_are_for_selection_not_gradient_or_independent_claim():
    train = [dict(sample_id=f'{s}-train-{i}', dataset=s) for s in 'abcd' for i in range(2)]
    val = [dict(sample_id=f'{s}-val', dataset=s) for s in 'abcd']
    roles = {role: {'allowed_source_queries': {s: 2 for s in 'abcd'}} for role in ['full', 'fold_0']}
    plan = prepare.leave_source_validation(train, val, roles)
    assert plan['heldout_validation_labels_allowed_for_cv_selection'] is True
    assert plan['cv_score_is_independent_final_evaluation'] is False
    a = plan['definitions']['a']
    assert a['gradient_train_n'] == 6 and a['cv_validation_ids'] == ['a-val']
    assert all(not sid.startswith('a-') for sid in a['gradient_train_ids'])
    assert a['frozen_proposer_has_seen_heldout_source'] is True
    assert a['original_validation_was_used_for_upstream_checkpoint_selection'] is True


def test_sealed_identity_and_byte_copy_fail_closed(tmp_path):
    source = tmp_path / 'original.jsonl'
    source.write_bytes(b'{"original":  true}\n')
    digest = prepare.sha(source)
    seal = {'files': {str(source): digest}, 'training_evidence': {str(source): digest}}
    identity = prepare.seal_inventory(seal)
    assert prepare.authenticate(source, identity)['matches_v12_seal'] is True
    target = tmp_path / 'v13.jsonl'
    prepare.copy_frozen_bytes(source, target)
    assert target.read_bytes() == source.read_bytes()
    source.write_bytes(b'{"changed": true}\n')
    with pytest.raises(AssertionError, match='Sealed V12 input changed'):
        prepare.authenticate(source, identity)
    with pytest.raises(AssertionError, match='Existing V13 retained cohort differs'):
        prepare.copy_frozen_bytes(source, target)


def test_primary_and_supplement_pairwise_near_overlap_fails():
    a = row('primary', 'screenspot_pro')
    b = row('supplement', 'screenspot_v2', int(a['dhash256'], 16) ^ 1)
    with pytest.raises(AssertionError, match='Historical/new near reuse'):
        prepare.verify_independence([], [], [a], [b], set(), set())
