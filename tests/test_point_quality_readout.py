"""Small CPU checks for isolated fixed-point quality supervision."""
import copy
import inspect

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from clickagent.attention_field_ranker import AttentionFieldRanker, FieldRankerConfig, make_field_batch
from clickagent.candidate_ranker import ranker_loss
from clickagent.point_quality_readout import (FROZEN_POINT_FEATURE_DIM,
    FROZEN_POINT_FEATURE_NAMES, PointQualityReadout, quality_loss)
from test_attention_field_ranker import field_record


def _model():
    return AttentionFieldRanker(FieldRankerConfig(4), np.ones(96)/96)


def _point_features(count, seed=9):
    generator = torch.Generator().manual_seed(seed)
    return torch.rand((count, FROZEN_POINT_FEATURE_DIM), generator=generator)


def test_quality_gradients_are_isolated_from_core_and_frozen_inputs():
    torch.manual_seed(10)
    inputs, _ = make_field_batch([field_record()], 'cpu')
    model = PointQualityReadout(_model(), quality_width=8)
    point = _point_features(len(inputs['image_hidden'])).requires_grad_()
    inputs['query_hidden'] = inputs['query_hidden'].float().requires_grad_()
    output = model(frozen_point_features=point, **inputs)
    target = torch.arange(len(point)) % 2
    loss = quality_loss(output['point_quality_logits'], target, inputs['candidate_lengths'])
    loss.backward()
    assert torch.isfinite(loss)
    assert all(parameter.grad is None for parameter in model.core.parameters())
    assert point.grad is None and inputs['query_hidden'].grad is None
    grads = [parameter.grad for parameter in model.quality.parameters()]
    assert all(gradient is not None and torch.isfinite(gradient).all() for gradient in grads)
    assert any(bool(gradient.abs().sum() > 0) for gradient in grads)


def test_wrapper_preserves_rng_rank_outputs_and_core_training_gradients():
    torch.manual_seed(29)
    core = _model()
    reference = copy.deepcopy(core)
    rng = torch.get_rng_state().clone()
    wrapper = PointQualityReadout(core, quality_width=8)
    assert torch.equal(rng, torch.get_rng_state())
    inputs, labels = make_field_batch([field_record(3), field_record(7)], 'cpu')
    features = _point_features(len(inputs['image_hidden']))
    torch.manual_seed(31)
    original = reference(**inputs)
    original_loss, _ = ranker_loss(original, **labels)
    original_loss.backward()
    original_rng = torch.get_rng_state().clone()
    torch.manual_seed(31)
    output = wrapper(frozen_point_features=features, **inputs)
    wrapped_loss, _ = ranker_loss(output, **labels)
    quality = quality_loss(output['point_quality_logits'], torch.zeros(len(features)), inputs['candidate_lengths'])
    (wrapped_loss+quality).backward()
    assert torch.equal(original_rng, torch.get_rng_state())
    for key in ('scores', 'null_scores', 'child_logits', 'offsets'):
        torch.testing.assert_close(output[key], original[key], atol=0, rtol=0)
    for parameter, expected in zip(core.parameters(), reference.parameters()):
        if expected.grad is None:
            assert parameter.grad is None
        else:
            torch.testing.assert_close(parameter.grad, expected.grad, atol=0, rtol=0)


def test_quality_batching_is_isolated_between_screenshots():
    torch.manual_seed(6)
    model = PointQualityReadout(_model(), quality_width=8).eval()
    records = [field_record(4), field_record(7)]
    inputs, _ = make_field_batch(records, 'cpu')
    features = _point_features(len(inputs['image_hidden']))
    together = model(frozen_point_features=features, **inputs)
    cursor = 0
    for record in records:
        one, _ = make_field_batch([record], 'cpu')
        count = len(record['parent_indices'])
        separate = model(frozen_point_features=features[cursor:cursor+count], **one)
        for key in ('point_quality_logits', 'combined_scores', 'quality_only_scores'):
            torch.testing.assert_close(together[key][cursor:cursor+count], separate[key], atol=2e-5, rtol=2e-5)
        cursor += count


def test_score_combination_and_label_free_forward_contract():
    assert len(FROZEN_POINT_FEATURE_NAMES) == FROZEN_POINT_FEATURE_DIM == 13
    assert 'labels_fixed_point_hit' not in inspect.signature(PointQualityReadout.forward).parameters
    inputs, _ = make_field_batch([field_record()], 'cpu')
    model = PointQualityReadout(_model(), quality_width=8).eval()
    features = _point_features(len(inputs['image_hidden']))
    output = model(frozen_point_features=features, **inputs)
    torch.testing.assert_close(output['combined_scores'], output['scores']+F.logsigmoid(output['point_quality_logits']))
    torch.testing.assert_close(output['quality_only_scores'], output['point_quality_logits'])
    assert torch.isfinite(output['combined_scores']).all()
    assert bool((output['combined_scores'] <= output['scores']).all())
    for label_name in ('labels_fixed_point_hit', 'label_fixed_point_hit', 'label_hard_negative'):
        with pytest.raises(TypeError, match='Supervision is loss-only'):
            model(frozen_point_features=features, **{label_name: torch.ones(len(features))}, **inputs)


def test_quality_loss_balances_classes_then_screenshots_and_handles_no_positive():
    logits = torch.tensor([2., -1., .5, -.5, 4., -4., 0.], requires_grad=True)
    targets = torch.tensor([1., 0., 0., 0., 0., 0., 1.])
    loss = quality_loss(logits, targets, [4, 2, 1])
    elements = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    expected = torch.stack(((elements[0]+elements[1:4].mean())*.5,
                            elements[4:6].mean(), elements[6])).mean()
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert torch.isfinite(logits.grad).all()
    for label in (0., 1.):
        values = torch.tensor([-1000., 1000.], requires_grad=True)
        value = quality_loss(values, torch.full((2,), label), [2])
        value.backward()
        assert torch.isfinite(value) and torch.isfinite(values.grad).all()


def test_quality_input_validation_rejects_invalid_partition_or_targets():
    with pytest.raises(ValueError, match='partition'):
        quality_loss(torch.zeros(3), torch.zeros(3), [2])
    with pytest.raises(ValueError, match='binary'):
        quality_loss(torch.zeros(2), torch.tensor([0., .5]), [2])
    with pytest.raises(ValueError, match='positive integer'):
        quality_loss(torch.zeros(2), torch.zeros(2), [0, 2])
    model = PointQualityReadout(_model(), quality_width=8).eval()
    inputs, _ = make_field_batch([field_record()], 'cpu')
    with pytest.raises(ValueError, match='candidate,13'):
        model(frozen_point_features=torch.zeros(len(inputs['image_hidden']), 12), **inputs)
    features = _point_features(len(inputs['image_hidden']))
    features[0, 0] = float('nan')
    with pytest.raises(ValueError, match='finite'):
        model(frozen_point_features=features, **inputs)
