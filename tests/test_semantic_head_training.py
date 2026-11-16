"""CPU checks for training order, hard examples and label-free v7 batching."""
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from semantic_head_data import make_batch, subset_record
from train_semantic_head_experiment import epoch_order, hard_negative_loss

from clickagent.attention_field_ranker import build_attention_field
from test_candidate_ranker import record


def _records_for_order():
    return [dict(case=dict(pixel_sha256=group))
            for group, size in [('same_a', 4), ('same_b', 3), ('same_c', 5), ('same_d', 2)]
            for _ in range(size)]


def _record(seed):
    result = record(seed)
    rng = np.random.default_rng(seed+400)
    result['attention_field'] = build_attention_field(
        rng.random((3, 96, 6), dtype=np.float32), result['parent_indices'])
    n = len(result['parent_indices'])
    result['frozen_points'] = np.arange(n*2, dtype=np.int64).reshape(n, 2)
    result['frozen_point_features'] = np.arange(n*13, dtype=np.float32).reshape(n, 13).astype(np.float16)
    result['label_fixed_point_hit'] = (np.arange(n) % 2).astype(np.float32)
    result['label_hard_negative'] = ((result['label_area'].sum(-1) == 0) & (np.arange(n) % 2 == 0)).astype(np.float32)
    return result


def test_random_order_is_reproducible_unique_and_respects_query_budget():
    train = _records_for_order()
    first = epoch_order(train, np.random.default_rng(7), 'random', budget=9)
    second = epoch_order(train, np.random.default_rng(7), 'random', budget=9)
    other = epoch_order(train, np.random.default_rng(8), 'random', budget=9)
    np.testing.assert_array_equal(first, second)
    assert len(first) == len(set(first.tolist())) == 9
    assert not np.array_equal(first, other)
    assert set(first).issubset(range(len(train)))


def test_grouped_hard_order_keeps_same_screenshot_contiguous_without_duplicates():
    train = _records_for_order()
    for budget in (9, len(train), len(train)+5):
        first = epoch_order(train, np.random.default_rng(17), 'grouped_hard', budget)
        second = epoch_order(train, np.random.default_rng(17), 'grouped_hard', budget)
        np.testing.assert_array_equal(first, second)
        assert len(first) == len(set(first.tolist())) == min(budget, len(train))
        groups = [train[index]['case']['pixel_sha256'] for index in first]
        for group in set(groups):
            positions = [i for i, value in enumerate(groups) if value == group]
            assert positions[-1]-positions[0]+1 == len(positions)
    full = epoch_order(train, np.random.default_rng(4), 'grouped_hard', len(train))
    assert sorted(full.tolist()) == list(range(len(train)))


def test_grouped_hard_and_random_have_the_same_selected_query_set():
    train = _records_for_order()
    for seed in range(12):
        for budget in (1, 7, 9, len(train), len(train)+5):
            a = epoch_order(train, np.random.default_rng(seed), 'random', budget)
            b = epoch_order(train, np.random.default_rng(seed), 'grouped_hard', budget)
            assert sorted(a.tolist()) == sorted(b.tolist())


def test_hard_negative_loss_falls_when_positive_score_increases():
    scores = torch.tensor([0., 1., -.5, .2, 1.5, -1.], requires_grad=True)
    area = torch.zeros(6, 4); area[0, 0] = 1; area[3, 1] = .4
    hard = torch.tensor([0., 1., 1., 0., 1., 1.])
    output = dict(scores=scores, candidate_lengths=[3, 3])
    loss = hard_negative_loss(output, area, hard)
    improved = scores.detach().clone(); improved[[0, 3]] += 2
    after = hard_negative_loss(dict(scores=improved, candidate_lengths=[3, 3]), area, hard)
    assert torch.isfinite(loss) and 0 < after < loss
    loss.backward()
    assert torch.isfinite(scores.grad).all()
    assert scores.grad[0] < 0 and scores.grad[3] < 0
    assert bool((scores.grad[[1, 2, 4, 5]] > 0).all())


def test_hard_negative_loss_handles_cases_missing_positive_or_negative():
    scores = torch.tensor([1000., -1000., 0., 2., -2., 1.], requires_grad=True)
    area = torch.zeros(6, 4); area[:3, 0] = 1
    hard = torch.tensor([0., 0., 0., 1., 1., 1.])
    loss = hard_negative_loss(dict(scores=scores, candidate_lengths=[3, 3]), area, hard)
    assert torch.isfinite(loss) and float(loss) == 0
    loss.backward()
    torch.testing.assert_close(scores.grad, torch.zeros_like(scores))


def test_hard_negative_loss_is_balanced_over_eligible_cases_without_cross_image_pairs():
    scores = torch.tensor([0., 1., 9., 4., 2., -20.])
    area = torch.zeros(6, 4); area[0, 0] = 1; area[3, 0] = 1
    hard = torch.tensor([0., 1., 0., 0., 1., 0.])
    loss = hard_negative_loss(dict(scores=scores, candidate_lengths=[3, 3]), area, hard)
    expected = (F.softplus(torch.tensor(.5+1.-0.))+F.softplus(torch.tensor(.5+2.-4.)))/2
    torch.testing.assert_close(loss, expected)
    changed = scores.clone(); changed[[2, 5]] = torch.tensor([-1000., 1000.])
    torch.testing.assert_close(loss, hard_negative_loss(dict(scores=changed, candidate_lengths=[3, 3]), area, hard))


def test_subset_record_preserves_alignment_of_frozen_point_and_supervision_arrays():
    original = _record(4)
    keep = np.array([True, False, True, False, False, True])
    subset = subset_record(original, keep)
    for name in ('parent_indices', 'image_hidden', 'vision_hidden', 'attention_field',
                 'candidate_features', 'label_area', 'label_offsets', 'frozen_points',
                 'frozen_point_features', 'label_fixed_point_hit', 'label_hard_negative'):
        np.testing.assert_array_equal(subset[name], original[name][keep])
    source_keep = keep[original['source_candidate']]
    mapping = np.cumsum(keep)-1
    np.testing.assert_array_equal(subset['source_candidate'], mapping[original['source_candidate'][source_keep]])
    assert len(original['parent_indices']) == 6
    assert subset['metadata'] == original['metadata']


def test_make_batch_inference_has_no_ground_truth_and_ignores_changed_labels():
    records = [_record(4), _record(7)]
    inputs, target = make_batch(records, 'cpu', labels=False)
    assert target is None
    assert all(not name.startswith('label') for name in inputs)
    assert not ({'case', 'metadata', 'geometry', 'target_geometry', 'frozen_points',
                 'label_fixed_point_hit', 'label_hard_negative'} & inputs.keys())
    changed = []
    for item in records:
        modified = dict(item)
        for name in ('label_area', 'label_offsets', 'label_fixed_point_hit', 'label_hard_negative'):
            modified[name] = np.ones_like(item[name])*7
        changed.append(modified)
    after, no_target = make_batch(changed, 'cpu', labels=False)
    assert no_target is None and inputs.keys() == after.keys()
    for name, value in inputs.items():
        if isinstance(value, torch.Tensor): torch.testing.assert_close(value, after[name])
        else: assert value == after[name]
    _, supervised = make_batch(records, 'cpu', labels=True)
    assert set(supervised) == {'label_area', 'label_offsets', 'label_fixed_point_hit', 'label_hard_negative'}
    for name, value in supervised.items():
        np.testing.assert_array_equal(value.numpy(), np.concatenate([item[name] for item in records]))
